import multiprocessing as mp
from icecream import ic
import librosa
import sounddevice as sd
import torch
import torchaudio
import whisper
import keyboard
from collections import deque
import numpy as np
from queue import Empty
import threading
import os
from dotenv import load_dotenv
from typing import Optional, Callable
import soundfile as sf
import time
from pyannote.audio import Pipeline, Inference
from qdrant_client import QdrantClient, models

load_dotenv()

# Get environment variables
huggingface = os.getenv("HUGGING_FACE")

# Constants
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
MAX_BUFFER_SIZE = 300  # max 150 seconds of audio
WHISPER_MODEL = "tiny"  # Use tiny model for faster transcription

# Global state for key detection
key_pressed = False
listening = False


def audio_capture_process(audio_queue, stop_event):
    """
    Process that captures audio from the microphone in chunks.
    Runs continuously and adds audio chunks to the queue when listening is active.
    """
    print("[Audio Capture] Starting audio capture process...")
    chunk_count = 0
    
    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype="float32",
        ) as stream:
            print(f"[Audio Capture] Stream opened successfully")
            while not stop_event.is_set():
                audio_chunk, overflow = stream.read(CHUNK_SIZE)
                if not stop_event.is_set():
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        print(f"[Audio Capture] Captured {chunk_count} chunks")
                    
                if overflow:
                    print("[Audio Capture] Audio buffer overflow - some audio may have been lost")
    except Exception as e:
        print(f"[Audio Capture] Error in audio capture: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Audio Capture] Audio capture process stopped")


def audio_buffer_process(
    audio_queue,
    command_queue,
    stop_event,
    listening_event,
):
    """
    Process that manages the audio buffer.
    Accumulates audio chunks when listening is active.
    Sends complete audio data to transcription when listening stops.
    """
    print("[Buffer] Starting audio buffer process...")
    
    audio_buffer = []
    was_listening = False
    chunk_count = 0
    
    try:
        while not stop_event.is_set():
            try:
                audio_chunk = audio_queue.get(timeout=0.5)
                chunk_count += 1
                
                if listening_event.is_set():
                    # Add chunk to buffer while listening
                    audio_buffer.append(audio_chunk)
                    was_listening = True
                    if len(audio_buffer) % 5 == 0:
                        print(f"[Buffer] Buffering... {len(audio_buffer)} chunks")
                        
            except Empty:
                # Check if listening just stopped and we have audio to process
                if was_listening and not listening_event.is_set() and len(audio_buffer) > 0:
                    # Send accumulated audio to transcription
                    print(f"[Buffer] RELEASE detected! Sending {len(audio_buffer)} chunks ({len(audio_buffer) * CHUNK_DURATION:.1f}s) to transcription...")
                    full_audio = np.concatenate(audio_buffer, axis=0)
                    print(f"[Buffer] Full audio shape: {full_audio.shape}")
                    command_queue.put(("transcribe", full_audio))
                    audio_buffer.clear()
                    was_listening = False
                    
    except Exception as e:
        print(f"[Buffer] Error in audio buffer process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Buffer] Audio buffer process stopped")


def speaker_embedding_process(audio_data, sample_rate):
    """
    Extract speaker embedding from audio data using pyannote.
    Returns the embedding vector.
    """
    # This function now expects a pre-loaded pyannote `pipeline` to be passed
    # so that the heavy model is not reloaded on every call. Keep this
    # function focused on converting audio and extracting embeddings.
    try:
        # Minimum duration check (avoid too-short audio)
        min_seconds = 0.3
        if len(audio_data) < int(min_seconds * sample_rate):
            print(f"[Speaker] Audio too short for embedding ({len(audio_data)/sample_rate:.2f}s)")
            return None

        # If caller passed a pipeline object in place of audio_data (old API), handle gracefully
        print("[Speaker] Extracting speaker embedding...")
        # The caller should pass a pipeline and audio_data tuple, but to keep
        # compatibility we expect the caller to call the pipeline directly.
        # This function will not load the pipeline itself.
        print("[Speaker] speaker_embedding_process called without pipeline - returning None")
        return None
    except Exception as e:
        print(f"[Speaker] Error in speaker embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_audio_for_pyannote(audio_data, sample_rate):
    """
    Convert arbitrary audio into a clean, normalized 16 kHz waveform
    safe for pyannote speaker embedding.

    Returns:
        tensor (torch.Tensor): shape (1, time), float32, CPU
    """

    print(f"[Speaker PREP] Input: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min():.4f}, max={audio_data.max():.4f}, mean={audio_data.mean():.4f}")

    # ---------- 1. Ensure float32 ----------
    audio_data = audio_data.astype(np.float32)
    print(f"[Speaker PREP] After float32: min={audio_data.min():.4f}, max={audio_data.max():.4f}")

    # ---------- 2. Remove NaN/Inf ----------
    audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[Speaker PREP] After nan_to_num: min={audio_data.min():.4f}, max={audio_data.max():.4f}")

    # ---------- 3. Make mono (if multichannel) ----------
    if audio_data.ndim == 2:
        print(f"[Speaker PREP] Converting from stereo to mono...")
        audio_data = librosa.to_mono(audio_data.T)
        print(f"[Speaker PREP] After to_mono: shape={audio_data.shape}, min={audio_data.min():.4f}, max={audio_data.max():.4f}")

    # ---------- 4. Resample to 16k (ONLY if needed) ----------
    if sample_rate != 16000:
        print(f"[Speaker PREP] Resampling from {sample_rate} to 16000...")
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        print(f"[Speaker PREP] After resample: shape={audio_data.shape}, min={audio_data.min():.4f}, max={audio_data.max():.4f}")
        sample_rate = 16000
    else:
        print(f"[Speaker PREP] Already at 16kHz, skipping resample")

    # ---------- 5. NO normalization - keep audio as is ----------
    print(f"[Speaker PREP] Final audio stats: min={audio_data.min():.4f}, max={audio_data.max():.4f}, mean={audio_data.mean():.4f}")

    # ---------- 6. Final tensor ----------
    tensor = torch.from_numpy(audio_data).float().unsqueeze(0).contiguous().cpu()
    print(f"[Speaker PREP] Tensor created: shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

    return tensor, sample_rate

def transcription_process(command_queue, result_queue, stop_event):
    """
    Process that handles transcription of audio.
    Receives audio data from the buffer process and transcribes it using Whisper.
    """
    print("[Transcription] Starting transcription process...")
    print("[Transcription] Loading Whisper model...")
    
    # Force CPU for Whisper to avoid MPS sparse tensor issues
    device = torch.device("cpu")
    print(f"[Transcription] Using device: {device}")
    client = QdrantClient("http://localhost:6333")
    collection_name = "voicesinmyhead"


    
    try:
        model = whisper.load_model(WHISPER_MODEL, device=device)
        print(f"[Transcription] Whisper model ({WHISPER_MODEL}) loaded on {device}")
        # Load speaker embedding model for direct embedding extraction
        try:
            print("[Transcription] Loading speaker embedding model...")
            # from pyannote.audio import Inference, Model
            speaker_embedding_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                token=huggingface,
            )

            # Force pyannote pipeline to CPU to avoid device mismatch between
            # in-memory waveform (CPU tensor) and model device which can cause
            # unexpected empty/NaN embeddings on some backends (MPS/CUDA).
            speaker_device = torch.device("cpu")
            try:
                speaker_embedding_model.to(speaker_device)
                print(f"[Transcription] Speaker embedding model loaded on {speaker_device}")
            except Exception:
                # Some Pipeline implementations may not support .to(); ignore if so
                print("[Transcription] Warning: could not set pipeline device to CPU; proceeding (pipeline may handle device itself)")
        except Exception as e:
            print(f"[Transcription] Could not load speaker embedding model: {e}")
            import traceback
            traceback.print_exc()
            speaker_embedding_model = None
        
        while not stop_event.is_set():
            try:
                print("[Transcription] Waiting for audio commands...")
                command_type, audio_data = command_queue.get(timeout=1.0)
                print(f"[Transcription] Got command from queue: {command_type}")
                
                if command_type == "transcribe":
                    print(f"[Transcription] === STARTING TRANSCRIPTION ===")
                    print(f"[Transcription] Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                    print(f"[Transcription] Processing audio ({len(audio_data)} samples, {len(audio_data)/SAMPLE_RATE:.1f}s)...")
                    
                    # Flatten if needed (remove channel dimension)
                    if audio_data.ndim > 1:
                        print(f"[Transcription] Flattening audio from shape {audio_data.shape}")
                        audio_data = audio_data.flatten()
                    
                    # Keep original audio for speaker embedding (before normalization)
                    audio_for_speaker = audio_data.copy()
                    print(f"[Transcription] SAVED audio_for_speaker: shape={audio_for_speaker.shape}, dtype={audio_for_speaker.dtype}, min={audio_for_speaker.min():.4f}, max={audio_for_speaker.max():.4f}")
                    
                    # Normalize audio to [-1, 1] range for Whisper
                    audio_data = audio_data.astype(np.float32)
                    max_val = np.abs(audio_data).max()
                    if max_val > 0:
                        audio_data = audio_data / max_val
                    
                    print(f"[Transcription] Normalized audio - min: {audio_data.min():.3f}, max: {audio_data.max():.3f}")
                    
                    # Transcribe with language specification for faster processing
                    try:
                        print("[Transcription] Calling model.transcribe()...")
                        result = model.transcribe(audio_data, language="en", verbose=False)
                        print("[Transcription] Transcription completed!")
                        transcript = result.get("text", "").strip()
                        
                        # Extract speaker embedding using ORIGINAL audio (before Whisper normalization)
                        speaker_name = "Unknown"
                        try:
                            if speaker_embedding_model is None:
                                print("[Speaker] No speaker embedding model available; skipping speaker identification")
                                speaker_embedding = None
                            else:
                                print(f"[Speaker] Audio for embedding: shape={audio_for_speaker.shape}, dtype={audio_for_speaker.dtype}, min={audio_for_speaker.min():.4f}, max={audio_for_speaker.max():.4f}")
                                
                                # Prepare audio_for_speaker for in-memory pipeline call (no temp files)
                                try:
                                    af = audio_for_speaker
                                    # Ensure numpy
                                    if not isinstance(af, np.ndarray):
                                        af = np.asarray(af)

                                    # Mono conversion if needed
                                    if af.ndim == 2:
                                        # librosa.to_mono expects shape (n_channels, n_samples) when transposed
                                        try:
                                            af = librosa.to_mono(af.T)
                                        except Exception:
                                            af = af.mean(axis=1)

                                    # Ensure float32 and normalized to [-1, 1]
                                    if af.dtype.kind != 'f':
                                        af = af.astype(np.float32)
                                        # assume int16
                                        af = af / 32768.0
                                    else:
                                        af = af.astype(np.float32)
                                        if np.abs(af).max() > 1.5:
                                            # values appear as int-like, normalize
                                            af = af / 32768.0

                                    # Final check shape -> (channels, samples)
                                    if af.ndim == 1:
                                        np_wave = af[np.newaxis, :]
                                    elif af.ndim == 2:
                                        # assume (samples, channels) -> transpose
                                        if af.shape[0] < af.shape[1]:
                                            np_wave = af.T
                                        else:
                                            np_wave = af
                                    else:
                                        raise ValueError(f"Unexpected af ndim: {af.ndim}")

                                    # Create tensor (channels, samples), float32, CPU
                                    waveform = torch.from_numpy(np_wave).float().contiguous().cpu()
                                    print(f"[Speaker] Prepared waveform tensor: shape={waveform.shape}, dtype={waveform.dtype}, device={waveform.device}")

                                    # Call pipeline with in-memory waveform dict
                                    print(f"[Speaker] Calling diarization pipeline (in-memory)...")
                                    diarization = speaker_embedding_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
                                    print(f"[Speaker] Diarization output type: {type(diarization)}")

                                    # Extract speaker embeddings robustly
                                    embedding = None
                                    if hasattr(diarization, 'speaker_embeddings'):
                                        se = diarization.speaker_embeddings
                                        try:
                                            length = len(se)
                                        except Exception:
                                            length = 1
                                        print(f"[Speaker] speaker_embeddings present, length={length}")
                                        if length > 0:
                                            embedding = se[0]
                                    else:
                                        print("[Speaker] No speaker_embeddings attribute on diarization output. Attributes:", dir(diarization))

                                    # Extract speaker embeddings robustly
                                    embedding = None
                                    if hasattr(diarization, 'speaker_embeddings'):
                                        se = diarization.speaker_embeddings
                                        try:
                                            length = len(se)
                                        except Exception:
                                            length = 1
                                        print(f"[Speaker] speaker_embeddings present, length={length}")
                                        
                                        # Print diarization timeline for debugging
                                        try:
                                            timeline = diarization.get_timeline() if hasattr(diarization, 'get_timeline') else None
                                            print(f"[Speaker] Diarization timeline: {timeline}")
                                        except Exception as e:
                                            print(f"[Speaker] Could not get timeline: {e}")
                                        
                                        if length > 0:
                                            embedding = se[0]
                                    else:
                                        print("[Speaker] No speaker_embeddings attribute on diarization output. Attributes:", dir(diarization))

                                    if embedding is not None:
                                        emb = np.array(embedding, dtype=np.float32).reshape(-1)
                                        print(f"[Speaker] Embedding shape: {emb.shape}, min={np.nanmin(emb):.6f}, max={np.nanmax(emb):.6f}")
                                        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
                                        print(f"[Speaker] Embedding after nan_to_num: min={np.min(emb):.6f}, max={np.max(emb):.6f}")
                                        
                                        # If diarization produced all zeros (no segments detected), fallback to direct embedding extraction
                                        if np.max(np.abs(emb)) < 1e-6:
                                            print("[Speaker] Diarization produced zero/NaN embeddings (no segments detected); attempting fallback direct embedding extraction...")
                                            try:
                                                from pyannote.audio import Model, Inference
                                                # Load the embedding extractor model directly
                                                embedding_model = Model.from_pretrained(
                                                    "pyannote/embedding",
                                                    token=huggingface,
                                                )
                                                embedding_model = embedding_model.to(torch.device("cpu"))
                                                inference = Inference(embedding_model, window="whole")
                                                
                                                # Extract embedding directly (no diarization segmentation)
                                                # Inference expects AudioFile; create a minimal dict-like wrapper
                                                class WaveformFile:
                                                    def __init__(self, wf, sr):
                                                        self.waveform = wf
                                                        self.sample_rate = sr
                                                
                                                wav_file = WaveformFile(waveform, SAMPLE_RATE)
                                                direct_emb = inference(wav_file)
                                                
                                                if direct_emb is not None:
                                                    direct_emb_arr = np.array(direct_emb, dtype=np.float32).reshape(-1)
                                                    print(f"[Speaker] Fallback direct embedding shape: {direct_emb_arr.shape}, min={np.nanmin(direct_emb_arr):.6f}, max={np.nanmax(direct_emb_arr):.6f}")
                                                    direct_emb_arr = np.nan_to_num(direct_emb_arr, nan=0.0, posinf=0.0, neginf=0.0)
                                                    if np.max(np.abs(direct_emb_arr)) > 1e-6:
                                                        print("[Speaker] Fallback direct embedding is valid; using it")
                                                        emb = direct_emb_arr
                                                    else:
                                                        print("[Speaker] Fallback direct embedding also zero; will use diarization result")
                                                else:
                                                    print("[Speaker] Fallback Inference returned None")
                                            except Exception as fallback_err:
                                                print(f"[Speaker] Fallback direct embedding extraction failed: {fallback_err}")
                                                import traceback
                                                traceback.print_exc()
                                        
                                        speaker_embedding = emb.tolist()
                                    else:
                                        print("[Speaker] No valid embedding extracted from diarization output")
                                        speaker_embedding = None
                                except Exception as pipeline_err:
                                    print(f"[Speaker] In-memory pipeline/embedding extraction failed: {pipeline_err}")
                                    import traceback
                                    traceback.print_exc()


                            if speaker_embedding:
                                vec = speaker_embedding
                                ic(vec, collection_name)
                                print(f"[Speaker] Querying Qdrant with vector length {len(vec)}")
                                search_results = client.query_points(collection_name=collection_name, query=vec, limit=1)
                                if search_results:
                                    top = search_results.points[0]
                                    payload = getattr(top, 'payload', None) or {}
                                    speaker_name = payload.get('speaker', 'Unknown')
                                    print(f"[Speaker] Matched: {speaker_name} (score={getattr(top, 'score', None)})")
                                else:
                                    print("[Speaker] No matches returned from Qdrant")
                        except Exception as e:
                            print(f"[Speaker] Could not query Qdrant or extract embedding: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        if transcript:
                            print(f"\n{'='*60}")
                            print(f"SPEAKER: {speaker_name}")
                            print(f"TRANSCRIPTION: {transcript}")
                            print(f"{'='*60}\n")
                            result_queue.put(("transcript", transcript, speaker_name))
                        else:
                            print("[Transcription] No speech detected")
                            result_queue.put(("transcript", None, speaker_name))
                            
                    except Exception as transcribe_error:
                        print(f"[Transcription] Transcription error: {transcribe_error}")
                        import traceback
                        traceback.print_exc()
                        result_queue.put(("transcript", None, "Unknown"))
                        
            except Empty:
                print("[Transcription] Queue timeout (no commands)")
                continue
                
    except Exception as e:
        print(f"[Transcription] Error in transcription process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Transcription] Transcription process stopped")


def key_listener_thread(listening_event, stop_event):
    """
    Thread that listens for keyboard input.
    When a specified key is held, sets the listening_event.
    When released, clears the listening_event.
    Uses polling approach for macOS compatibility.
    """
    # Define the key to listen for (Space key by default)
    LISTEN_KEY = "space"
    
    print("[Listener] Starting keyboard listener thread...")
    print("[Listener] Listening for 'space' key press/release...")
    
    try:
        was_pressed = False
        
        while not stop_event.is_set():
            # Poll the keyboard state
            is_pressed = keyboard.is_pressed(LISTEN_KEY)
            
            # Detect press transition
            if is_pressed and not was_pressed:
                listening_event.set()
                print("[Listener] KEY PRESSED - Starting audio capture")
                was_pressed = True
            
            # Detect release transition
            elif not is_pressed and was_pressed:
                listening_event.clear()
                print("[Listener] KEY RELEASED - Stopping audio capture and transcribing...")
                was_pressed = False
            
            # Small sleep to avoid busy waiting
            time.sleep(0.01)
            
    except Exception as e:
        print(f"[Listener] Error in keyboard listener: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Listener] Keyboard listener thread stopped")


class PushToTalkPipeline:
    """
    Multiprocessing pipeline for push-to-talk voice command recognition.
    Handles audio capture, buffering, and transcription in separate processes.
    """
    
    def __init__(self, result_callback: Optional[Callable] = None):
        """
        Initialize the pipeline.
        
        Args:
            result_callback: Optional callback function to handle transcription results
        """
        self.result_callback = result_callback
        self.stop_event = mp.Event()
        self.listening_event = mp.Event()
        
        # Communication queues
        self.audio_queue = mp.Queue(maxsize=100)
        self.command_queue = mp.Queue(maxsize=50)
        self.result_queue = mp.Queue()
        
        # Processes
        self.processes = []
        self.listener_thread = None
        self.result_monitor_thread = None
    
    def start(self):
        """Start all processes and threads in the pipeline."""
        print("=" * 50)
        print("Starting Push-to-Talk Pipeline")
        print("=" * 50)
        print("Press and HOLD SPACE to record")
        print("Release SPACE to transcribe")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start audio capture process
        p_capture = mp.Process(
            target=audio_capture_process,
            args=(self.audio_queue, self.stop_event),
            daemon=True,
        )
        p_capture.start()
        self.processes.append(p_capture)
        
        # Start audio buffer process
        p_buffer = mp.Process(
            target=audio_buffer_process,
            args=(self.audio_queue, self.command_queue, self.stop_event, self.listening_event),
            daemon=True,
        )
        p_buffer.start()
        self.processes.append(p_buffer)
        
        # Start transcription process
        p_transcription = mp.Process(
            target=transcription_process,
            args=(self.command_queue, self.result_queue, self.stop_event),
            daemon=True,
        )
        p_transcription.start()
        self.processes.append(p_transcription)
        
        # Start keyboard listener thread
        self.listener_thread = threading.Thread(
            target=key_listener_thread,
            args=(self.listening_event, self.stop_event),
            daemon=True,
        )
        self.listener_thread.start()
        
        # Start result monitor thread
        self.result_monitor_thread = threading.Thread(
            target=self._monitor_results,
            daemon=True,
        )
        self.result_monitor_thread.start()
    
    def _monitor_results(self):
        """Monitor the result queue and call the callback function."""
        while not self.stop_event.is_set():
            try:
                result_tuple = self.result_queue.get(timeout=0.5)
                
                # Handle both old format (2 items) and new format (3 items with speaker)
                if len(result_tuple) == 2:
                    result_type, result_data = result_tuple
                    speaker_name = "Unknown"
                else:
                    result_type, result_data, speaker_name = result_tuple
                
                if result_type == "transcript" and result_data:
                    if self.result_callback:
                        self.result_callback(result_data, speaker_name)
                    else:
                        print(f"\n>>> SPEAKER: {speaker_name}")
                        print(f">>> TRANSCRIPT: {result_data}\n")
                        
            except Empty:
                continue
            except Exception as e:
                print(f"[Monitor] Error: {e}")
    
    def stop(self):
        """Stop all processes and threads in the pipeline."""
        print("\nShutting down pipeline...")
        self.stop_event.set()
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Wait for threads to finish
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2)
        
        if self.result_monitor_thread and self.result_monitor_thread.is_alive():
            self.result_monitor_thread.join(timeout=2)
        
        print("Pipeline shutdown complete")


def main():
    """Main entry point for the push-to-talk pipeline."""
    
    # Optional: Define a callback function to handle transcripts
    def on_transcript(transcript, speaker):
        print(f"\n{'='*50}")
        print(f"SPEAKER: {speaker}")
        print(f"VOICE COMMAND: {transcript}")
        print(f"{'='*50}\n")
    
    # Create and start the pipeline
    pipeline = PushToTalkPipeline(result_callback=on_transcript)
    
    try:
        pipeline.start()
        
        # Keep the main thread alive
        while True:
            pass
            
    except KeyboardInterrupt:
        print("\nInterrupt signal received")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method("spawn", force=True)
    main()
