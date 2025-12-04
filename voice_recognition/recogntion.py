import sounddevice as sd
import soundfile as sf
import torch
from qdrant_client import QdrantClient, models
from icecream import ic
import numpy as np
from pyannote.audio import Pipeline, Inference
import os
from dotenv import load_dotenv
load_dotenv()

# <<< NEW IMPORTS >>>
import whisper  # Whisper for offline transcription
# Make sure you have installed whisper (e.g. pip install openai-whisper) and ffmpeg on your system.

# Using os.getenv()
huggingface = os.getenv("HUGGING_FACE")

# Record 5 seconds audio
duration = 5  # seconds
sample_rate = 16000
filename = "temp.wav"

print("Recording for 5 seconds...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
sd.wait()
sf.write(filename, audio, sample_rate)
print("Audio saved to", filename)

# diarization pipeline (if you still want diarization)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=huggingface,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
pipeline.to(device)
audiopath = filename
speakername = audiopath.split(".")[0]
# run diarization and get one embedding per speaker
output = pipeline(audiopath, return_embeddings=True)
embedding = output.speaker_embeddings[0]
ic(embedding.shape)
embedding_vector = embedding.tolist()

# Setup Qdrant client
client = QdrantClient("http://localhost:6333")
collection_name = "voicesinmyhead"
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(embedding_vector), distance=models.Distance.COSINE),
    )
except Exception as e:
    print("Collection might already exist:", e)

query_result = client.query_points(
    collection_name=collection_name,
    query=embedding_vector,
    limit=5
)
ic(query_result)

# <<< NEW: Load Whisper model and transcribe audio >>>
print("Loading Whisper model...")
model = whisper.load_model("base")  # you can choose "tiny","base","small","medium","large" depending on your resources
print("Transcribing audio with Whisper...")
whisper_result = model.transcribe(audiopath)  # returns dict with 'text' key
transcript = whisper_result.get("text", "")

# Combine speaker recognition + transcription
print(f"Speaker embedding matched neighbors: {query_result.points[0].payload['speaker'] if query_result else 'No matches found'}")
print("Transcription of the recording:", transcript)


#/Users/msahitya14/dev/Minecraft-voice-commands/.venv/lib/python3.13/site-packages/lightning/fabric/utilities/cloud_io.py had to make change at line 76 
# return torch.load(
#             f,
#             map_location=map_location,  # type: ignore[arg-type]
#             weights_only=False,
#         )