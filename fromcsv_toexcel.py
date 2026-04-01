import pandas as pd
import json

df = pd.read_csv("rag_logs.csv")

# превращаем JSON строки в читаемый текст
df["retrieved_chunks"] = df["retrieved_chunks"].apply(lambda x: "\n\n".join(json.loads(x)))
df["reranked_chunks"] = df["reranked_chunks"].apply(lambda x: "\n\n".join(json.loads(x)))

df["retrieved_scores"] = df["retrieved_scores"].apply(lambda x: ", ".join(map(str, json.loads(x))))
df["reranked_scores"] = df["reranked_scores"].apply(lambda x: ", ".join(map(str, json.loads(x))))

# сохраняем в Excel
df.to_excel("rag_logs_clean.xlsx", index=False)