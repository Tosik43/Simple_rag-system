import json
import os
import csv

from datetime import datetime
from core.config import LOG_FILE

def save_to_csv(
    question,
    retrieved,
    reranked,
    answer,
    embed_time,
    search_time,
    rerank_time,
    gen_time
):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "retrieved_chunks",
                "retrieved_scores",
                "reranked_chunks",
                "reranked_scores",
                "answer",
                "embed_time",
                "search_time",
                "rerank_time",
                "generation_time"
            ])

        retrieved_texts = [r.payload["text"] for r in retrieved]
        retrieved_scores = [r.score for r in retrieved]

        reranked_texts = [r.payload["text"] for r in reranked]
        reranked_scores = [r.score for r in reranked]

        writer.writerow([
            datetime.utcnow().isoformat(),
            question,
            json.dumps(retrieved_texts, ensure_ascii=False),
            json.dumps(retrieved_scores),
            json.dumps(reranked_texts, ensure_ascii=False),
            json.dumps(reranked_scores),
            answer,
            embed_time,
            search_time,
            rerank_time,
            gen_time
        ])