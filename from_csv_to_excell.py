import pandas as pd
import json

# загружаем CSV
df = pd.read_csv("rag_logs.csv")

# === Преобразование JSON полей в читаемый текст ===

df["retrieved_chunks"] = df["retrieved_chunks"].apply(
    lambda x: "\n\n".join(json.loads(x))
)

df["reranked_chunks"] = df["reranked_chunks"].apply(
    lambda x: "\n\n".join(json.loads(x))
)

df["retrieved_scores"] = df["retrieved_scores"].apply(
    lambda x: ", ".join(map(str, json.loads(x)))
)

df["reranked_scores"] = df["reranked_scores"].apply(
    lambda x: ", ".join(map(str, json.loads(x)))
)

# === Работа с таймингами ===

time_cols = ["embed_time", "search_time", "rerank_time", "generation_time"]

for col in time_cols:
    if col in df.columns:
        df[col] = df[col].astype(float).round(3)

# общее время pipeline
if all(col in df.columns for col in time_cols):
    df["total_time"] = (
        df["embed_time"]
        + df["search_time"]
        + df["rerank_time"]
        + df["generation_time"]
    )

# === Дополнительная аналитика ===

# количество чанков
df["retrieved_chunks_count"] = df["retrieved_chunks"].apply(
    lambda x: len(x.split("\n\n"))
)

df["reranked_chunks_count"] = df["reranked_chunks"].apply(
    lambda x: len(x.split("\n\n"))
)

# === Сохраняем в Excel ===

df.to_excel("rag_logs_clean.xlsx", index=False)