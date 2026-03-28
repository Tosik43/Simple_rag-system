import json
import os
from datetime import datetime
from tqdm import tqdm

from main import retrieve
from reranker import rerank

# ================= CONFIG =================
INITIAL_K = 50   # сколько берем из retriever
FINAL_K = 10     # на чем считаем метрики

# =========================================


def evaluate(dataset_path, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = {
        "no_rerank": init_metrics(),
        "rerank": init_metrics()
    }

    for item in tqdm(data):
        question = item["question"]
        relevant = set(item["relevant_chunks"])

        # ========= 1. WITHOUT RERANK =========
        results = retrieve(question)
        retrieved_ids = extract_ids(results[:FINAL_K])
        update_metrics(metrics["no_rerank"], retrieved_ids, relevant)

        # ========= 2. WITH RERANK =========
        results = retrieve(question)[:INITIAL_K]
        reranked = rerank(question, results)

        reranked_ids = extract_ids(reranked[:FINAL_K])
        update_metrics(metrics["rerank"], reranked_ids, relevant)

    # ========= FINALIZE =========
    summary = {
        key: finalize_metrics(metrics[key], total)
        for key in metrics
    }

    # ========= PRINT =========
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for key in ["no_rerank", "rerank"]:
        print(f"\n--- {key.upper()} ---")
        print(f"Hit Rate@{FINAL_K}: {summary[key]['hit_rate']:.4f}")
        print(f"Recall@{FINAL_K}: {summary[key]['recall']:.4f}")
        print(f"MRR@{FINAL_K}: {summary[key]['mrr']:.4f}")
        print(f"Precision@{FINAL_K}: {summary[key]['precision']:.4f}")

    # ========= SAVE =========
    summary_file = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ================= HELPERS =================

def init_metrics():
    return {
        "hit": 0,
        "recall_sum": 0,
        "mrr_sum": 0,
        "precision_sum": 0
    }


def extract_ids(results):
    return [
        r.payload["chunk_id"]
        for r in results
        if "chunk_id" in r.payload
    ]


def update_metrics(m, retrieved_ids, relevant):
    retrieved_set = set(retrieved_ids)

    # ===== Hit =====
    if any(cid in relevant for cid in retrieved_ids):
        m["hit"] += 1

    # ===== Recall =====
    if len(relevant) > 0:
        recall = len(retrieved_set & relevant) / len(relevant)
        m["recall_sum"] += recall

    # ===== Precision =====
    if len(retrieved_ids) > 0:
        precision = len(retrieved_set & relevant) / len(retrieved_ids)
        m["precision_sum"] += precision

    # ===== MRR =====
    rr = 0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            rr = 1 / rank
            break
    m["mrr_sum"] += rr


def finalize_metrics(m, total):
    return {
        "hit_rate": m["hit"] / total,
        "recall": m["recall_sum"] / total,
        "precision": m["precision_sum"] / total,
        "mrr": m["mrr_sum"] / total
    }


# ================= RUN =================

if __name__ == "__main__":
    evaluate("test system/rag_dataset_real.json")