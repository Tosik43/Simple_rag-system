import pandas as pd
import json

df = pd.read_csv("rag_logs.csv")

# ========= CHUNKS =========

df["retrieved_chunks"] = df["retrieved_chunks"].fillna("").apply(
    lambda x: "\n\n".join(str(x).split(" ||| "))
)

df["reranked_chunks"] = df["reranked_chunks"].fillna("").apply(
    lambda x: "\n\n".join(str(x).split(" ||| "))
)

# ========= SCORES =========

def parse_scores(x):
    try:
        return ", ".join(map(str, json.loads(x)))
    except:
        return str(x)

df["retrieved_scores"] = df["retrieved_scores"].apply(parse_scores)

df["reranked_scores"] = df["reranked_scores"].apply(parse_scores)

# ========= TIMES =========

time_cols = [
    "embed_time",
    "search_time",
    "rerank_time",
    "generation_time"
]

for col in time_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).round(3)

# ========= TOTAL =========

if all(col in df.columns for col in time_cols):
    df["total_time"] = (
        df["embed_time"]
        + df["search_time"]
        + df["rerank_time"]
        + df["generation_time"]
    ).round(3)

# ========= COUNTS =========

df["retrieved_chunks_count"] = df["retrieved_chunks"].apply(
    lambda x: len(str(x).split("\n\n"))
)

df["reranked_chunks_count"] = df["reranked_chunks"].apply(
    lambda x: len(str(x).split("\n\n"))
)

# ========= EXPORT =========

df.to_excel(
    "rag_logs_clean.xlsx",
    index=False
)

print("Excel saved: rag_logs_clean.xlsx")