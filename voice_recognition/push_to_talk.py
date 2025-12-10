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
from pyannote.audio import Pipeline, Inference, Model
from qdrant_client import QdrantClient, models

import time



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
    Process that handles transcription + speaker identification.
    Uses Whisper for ASR and pyannote/embedding for speaker ID.
    """

    print("[Transcription] Starting process...")

    device = torch.device("cpu")
    SAMPLE_RATE = 16000
    collection_name = "speaker_embeddings"

    # -------------------- Load models --------------------

    print("[Transcription] Loading Whisper...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=device)

    print("[Speaker] Loading speaker embedding model...")
    embedding_model = Model.from_pretrained(
        "pyannote/embedding",
        token=huggingface,
    ).to(device)

    embedder = Inference(embedding_model, window="whole")

    print("[Qdrant] Connecting...")
    qdrant = QdrantClient("http://localhost:6333")

    print("[Transcription] Ready ✅")

    # -------------------- Main loop --------------------

    while not stop_event.is_set():
        try:
            command, audio = command_queue.get(timeout=1.0)

            if command != "transcribe":
                continue

            # -------------------- Prepare audio --------------------

            if audio.ndim > 1:
                audio = audio.flatten()

            audio = audio.astype(np.float32)

            # Normalize to [-1, 1] if needed
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            duration = len(audio) / SAMPLE_RATE
            print(f"[Audio] {duration:.2f}s")

            # -------------------- Transcription --------------------

            result = whisper_model.transcribe(
                audio,
                language="en",
                verbose=False
            )

            transcript = result.get("text", "").strip()

            # -------------------- Speaker embedding --------------------

            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)

            with torch.no_grad():
                embedding = embedder({
                    "waveform": waveform,
                    "sample_rate": SAMPLE_RATE
                })

            if isinstance(embedding, torch.Tensor):
                emb = embedding.detach().cpu().numpy().reshape(-1)
            else:
                emb = np.asarray(embedding, dtype=np.float32).reshape(-1)

            emb = np.nan_to_num(emb, nan=0.0)

            speaker_name = "Unknown"

            # ic(emb)

            # -------------------- Qdrant lookup --------------------

            if np.linalg.norm(emb) > 1e-6:
                hits = qdrant.query_points(
                    collection_name=collection_name,
                    query=emb.tolist(),
                    limit=1,
                )

                if hits and hits.points:
                    top = hits.points[0]
                    speaker_name = (
                        top.payload.get("speaker", "Unknown")
                        if top.payload else "Unknown"
                    )

            # -------------------- Output --------------------

            if transcript:
                print(f"\n{'=' * 50}")
                print(f"SPEAKER : {speaker_name}")
                print(f"TEXT    : {transcript}")
                print(f"{'=' * 50}\n")

            result_queue.put(("transcript", transcript or None, speaker_name))

        except Empty:
            continue
        except Exception as e:
            print(f"[Error] {e}")
            result_queue.put(("transcript", None, "Unknown"))

    print("[Transcription] Stopped.")


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
