import os
import json
from datasets import Dataset

# ====== 1. Настройка GigaChat ======
from langchain.chat_models import GigaChat
from ragas.llms import LangchainLLMWrapper

GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")

if not GIGACHAT_API_KEY:
    raise ValueError("Установи переменную окружения GIGACHAT_API_KEY")

giga = GigaChat(
    credentials=GIGACHAT_API_KEY,
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS"
)

llm = LangchainLLMWrapper(giga)

# ====== 2. Загрузка данных ======
DATA_PATH = "ragas_results_20260329_123601.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Проверка структуры
required_keys = {"question", "contexts", "answer", "ground_truth"}
for i, row in enumerate(data):
    if not required_keys.issubset(row.keys()):
        raise ValueError(f"Ошибка в строке {i}: нет нужных полей")

dataset = Dataset.from_list(data)

# ====== 3. Метрики ======
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
]

# ====== 4. Запуск оценки ======
from ragas import evaluate

print("🚀 Запуск оценки RAGAS...\n")

result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=llm,
)

# ====== 5. Вывод результатов ======
print("\n📊 Итоговые метрики:\n")
print(result)

# В pandas
df = result.to_pandas()

print("\n📄 Примеры строк:\n")
print(df.head())

# ====== 6. Сохранение ======
OUTPUT_PATH = "ragas_evaluation_results.csv"
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n💾 Результаты сохранены в {OUTPUT_PATH}")