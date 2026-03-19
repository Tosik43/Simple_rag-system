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

    pairs = [(query, r.payload["text"]) for r in results]

    scores = reranker.predict(
        pairs,
        batch_size=16
    )

    scored_results = list(zip(results, scores))

    scored_results.sort(key=lambda x: x[1], reverse=True)

    return [r for r, _ in scored_results[:top_k]]