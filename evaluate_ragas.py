import json
import pandas as pd
import time

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

# LLM
from langchain_community.llms import HuggingFacePipeline
from ragas.llms import LangchainLLMWrapper
from sentence_transformers import SentenceTransformer

# HF
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Embeddings
from ragas.embeddings import HuggingFaceEmbeddings


# =========================
# CONFIG
# =========================
INPUT_FILE = "evaluation_results/ragas_results_qwen3.json"
OUTPUT_FILE = "results_ragas.csv"

# 👉 МОДЕЛЬ ДЛЯ ОЦЕНКИ (можно слабее!)
LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

# 👉 EMBEDDINGS
EMBED_MODEL_NAME = "BAAI/bge-m3"


# =========================
# DATASET
# =========================
def load_dataset(path: str) -> Dataset:
    print(f"Loading dataset from {path}...")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for item in data:
        dataset_dict["question"].append(item["question"])
        dataset_dict["answer"].append(item["answer"])
        dataset_dict["contexts"].append(item["contexts"])
        dataset_dict["ground_truth"].append(item["ground_truth"])

    print(f"Loaded {len(data)} samples")
    return Dataset.from_dict(dataset_dict)


# =========================
# LLM (LOCAL)
# =========================
def get_llm():
    print(f"Loading local LLM: {LLM_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        temperature=0
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return LangchainLLMWrapper(llm)


# =========================
# EMBEDDINGS
# =========================
def get_embeddings():
    print(f"Loading embeddings: {EMBED_MODEL_NAME}")

    return HuggingFaceEmbeddings(
        model=EMBED_MODEL_NAME   # ✅ строка, не объект
    )


# =========================
# EVALUATION
# =========================
def run_evaluation(dataset: Dataset, llm):
    embeddings = get_embeddings()

    print("Starting evaluation...")
    print(f"Samples: {len(dataset)}")

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness(),
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    return result


# =========================
# SAVE
# =========================
def save_results(result, output_path: str):
    df = result.to_pandas()

    summary_mean = df.mean(numeric_only=True).to_frame(name="mean").T
    summary_median = df.median(numeric_only=True).to_frame(name="median").T
    summary = pd.concat([summary_mean, summary_median])

    df.to_csv(output_path, index=False, encoding="utf-8")

    summary_path = output_path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=True, encoding="utf-8")

    print("\n=== RESULTS ===")
    print(summary)


# =========================
# MAIN
# =========================
def main():
    print("\n=== RAGAS LOCAL EVALUATION ===\n")

    dataset = load_dataset(INPUT_FILE)

    llm = get_llm()

    start = time.time()

    result = run_evaluation(dataset, llm)

    print(f"\nDone in {time.time() - start:.1f}s")

    save_results(result, OUTPUT_FILE)


if __name__ == "__main__":
    main()