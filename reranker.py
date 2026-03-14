import torch
import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=DEVICE,
    max_length=512
)

print("Reranker loaded")


def rerank(query: str, results: list, top_k: int = 3):

    """
    results = list from Qdrant

    returns reranked results
    """

    if len(results) == 0:
        return []

    pairs = []

    for result in results:
        pairs.append((query, result.payload["text"]))

    scores = reranker.predict(
        pairs,
        batch_size=16
    )

    reranked = []

    for score, result in zip(scores, results):

        reranked.append({
            "score": float(score),
            "text": result.payload["text"],
            "source": result.payload["source"],
            "page": result.payload["page"]
        })

    reranked.sort(key=lambda x: x["score"], reverse=True)

    return reranked[:top_k]