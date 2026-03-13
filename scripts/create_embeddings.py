import os
import json
import numpy as np
import torch

from typing import List, Dict
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

CHUNKS_FILE = "data/chunks.json"
EMBEDDINGS_FILE = "data/embeddings.npy"
METADATA_FILE = "data/metadata.json"

BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> SentenceTransformer:

    print(f"Используется устройство: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Загрузка модели: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print("Модель загружена")

    return model


def load_chunks(path: str) -> List[Dict]:

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Chunks загружены: {len(chunks)}")

    return chunks


# проверка структуры chunks
def validate_chunks(chunks: List[Dict]):

    required_fields = {"chunk_id", "text", "source", "page"}

    for i, chunk in enumerate(chunks):

        missing = required_fields - chunk.keys()

        if missing:
            raise ValueError(
                f"Chunk {i} missing fields: {missing}"
            )


def create_embeddings(
    model: SentenceTransformer,
    chunks: List[Dict],
    batch_size: int
) -> np.ndarray:

    texts = [chunk["text"] for chunk in chunks]

    print("\nСоздание embeddings")

    # отключение градиентов
    with torch.no_grad():

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=DEVICE
        )

    embeddings = embeddings.astype(np.float16)

    print("Embeddings созданы")

    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.save(path, embeddings)

    print(f"Embeddings сохранены: {path}")


def save_metadata(chunks: List[Dict], path: str):

    metadata = []

    for chunk in chunks:

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"]
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata сохранена: {path}")


def main():

    print("=== EMBEDDING PIPELINE STARTED ===\n")

    chunks = load_chunks(CHUNKS_FILE)

    validate_chunks(chunks)

    model = load_model()

    embeddings = create_embeddings(
        model,
        chunks,
        batch_size=BATCH_SIZE
    )

    print("\nShape:", embeddings.shape)

    save_embeddings(embeddings, EMBEDDINGS_FILE)

    save_metadata(chunks, METADATA_FILE)

    print("\n=== ГОТОВО ===")

    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")


# =====================================

if __name__ == "__main__":
    main()