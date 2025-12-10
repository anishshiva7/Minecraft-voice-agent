import os
import uuid
import numpy as np
import sounddevice as sd
import torch

from dotenv import load_dotenv
from pyannote.audio import Model, Inference
from qdrant_client import QdrantClient, models

# --------------------------------------------------
# Config
# --------------------------------------------------

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE")

SAMPLE_RATE = 16000
RECORD_SECONDS = 5
COLLECTION_NAME = "speaker_embeddings"

SPEAKER_NAME = "pranay"  # change per enrollment

# --------------------------------------------------
# Device (CPU recommended for pyannote stability)
# --------------------------------------------------

device = torch.device("cpu")

# --------------------------------------------------
# Record audio (in memory)
# --------------------------------------------------

print(f"[Audio] Recording {RECORD_SECONDS}s... Speak now 🎙️")

audio = sd.rec(
    int(RECORD_SECONDS * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)

sd.wait()

audio = audio.squeeze()  # shape: (samples,)
print(f"[Audio] Recorded shape: {audio.shape}")

# Normalize safety
max_val = np.abs(audio).max()
if max_val > 0:
    audio = audio / max_val

# --------------------------------------------------
# Load speaker embedding model
# --------------------------------------------------

print("[Speaker] Loading embedding model...")

model = Model.from_pretrained(
    "pyannote/embedding",
    token=HUGGINGFACE_TOKEN,
).to(device)

embedder = Inference(model, window="whole")

# --------------------------------------------------
# Run embedding inference (IN MEMORY)
# --------------------------------------------------

waveform = torch.from_numpy(audio).unsqueeze(0).to(device)

with torch.no_grad():
    embedding = embedder({
        "waveform": waveform,
        "sample_rate": SAMPLE_RATE,
    })

# Handle numpy / torch return safely
if isinstance(embedding, torch.Tensor):
    emb = embedding.detach().cpu().numpy().reshape(-1)
else:
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)

emb = np.nan_to_num(emb, nan=0.0)

print(
    f"[Speaker] Embedding shape={emb.shape}, norm={np.linalg.norm(emb):.4f}"
)

# --------------------------------------------------
# Push to Qdrant
# --------------------------------------------------

client = QdrantClient("http://localhost:6333")

try:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(emb),
            distance=models.Distance.COSINE,
        ),
    )
    print("[Qdrant] Collection created")
except Exception:
    print("[Qdrant] Collection already exists")

speaker_id = str(uuid.uuid4())

client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=speaker_id,
            vector=emb.tolist(),
            payload={"speaker": SPEAKER_NAME},
        )
    ],
)

print(f"[Qdrant] ✅ Speaker '{SPEAKER_NAME}' enrolled")
