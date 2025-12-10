import torch
from pyannote.audio import Pipeline, Inference, Model
from icecream import ic
from qdrant_client import QdrantClient, models
import uuid
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file
import numpy as np

# Using os.getenv()
huggingface = os.getenv("HUGGING_FACE")


client = QdrantClient("http://localhost:6333")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# diarization pipeline (if you still want diarization)
embedding_model = Model.from_pretrained(
    "pyannote/embedding",
    token=huggingface,
).to(device)

embedder = Inference(embedding_model, window="whole")


audiopath = "pranay.wav"
speakername = audiopath.split(".")[0]
# run diarization and get one embedding per speaker

with torch.no_grad():
    embedding = embedder(audiopath)

if isinstance(embedding, torch.Tensor):
    emb = embedding.detach().cpu().numpy().reshape(-1)
else:
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)

emb = np.nan_to_num(emb, nan=0.0)

embedding_vector = emb.tolist()

speaker_id = str(uuid.uuid4())

collection_name = "speaker_embeddings"

try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(embedding_vector), distance=models.Distance.COSINE),
    )
except Exception as e:
    print("Collection might already exist:", e)

operation_info = client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=speaker_id,
            vector=embedding_vector,
            payload={"speaker": speakername},
        )
    ]
)
print("Upserted:", operation_info)