import os
import json
import hashlib
import numpy as np
import ollama
import streamlit as st
import csv
import time
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from reranker import rerank

# ================= CONFIG =================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE")
METADATA_FILE = os.getenv("METADATA_FILE")

TOP_K_RETRIEVE = 10
TOP_K_RERANK = 3
MIN_SIMILARITY_SCORE = 0.3

NO_ANSWER_MESSAGE = "Нет информации по этому вопросу."

LOG_FILE = "rag_logs.csv"

# ================= INIT =================
@st.cache_resource
def load_models():
    print("[INIT] Загрузка embedding модели...")
    t0 = time.time()
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"[INIT] Embedding модель загружена за {time.time() - t0:.2f} сек")

    print("[INIT] Подключение к Qdrant...")
    t1 = time.time()
    qdrant = QdrantClient(url=QDRANT_URL)
    print(f"[INIT] Подключение к Qdrant установлено за {time.time() - t1:.2f} сек")

    return embed_model, qdrant

EMBED_MODEL, qdrant = load_models()

# ================= FUNCTIONS =================

def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def save_to_csv(question, retrieved, reranked, answer):
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
                "answer"
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
            answer
        ])


def retrieve(query: str):
    print("\n[STEP 1] Векторизация запроса...")
    t0 = time.time()
    query_vector = EMBED_MODEL.encode(query, normalize_embeddings=True)
    print(f"[STEP 1] Готово за {time.time() - t0:.3f} сек")

    print("[STEP 2] Поиск в Qdrant...")
    t1 = time.time()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE
    )
    print(f"[STEP 2] Найдено {len(results.points)} чанков за {time.time() - t1:.3f} сек")

    return results.points


def build_context(results):
    print("[STEP 4] Сбор контекста...")
    t0 = time.time()

    context = ""
    sources = []

    for i, r in enumerate(results):
        text = r.payload["text"]
        source = r.payload["source"]
        page = r.payload["page"]

        context += f"""
Источник {i+1}
Документ: {source}
Страница: {page}
{text}
"""

        sources.append(f"{source} (стр. {page})")

    print(f"[STEP 4] Контекст собран за {time.time() - t0:.3f} сек")

    return context, sources


def generate(query, context):
    print("[STEP 5] Отправка в LLM...")
    t0 = time.time()

    prompt = f"""
Ты — AI ассистент университета.

Отвечай ТОЛЬКО на основе предоставленной информации (контекста).
Запрещено использовать знания вне контекста, даже если они кажутся очевидными.

---

## АНАЛИЗ КОНТЕКСТА

Всегда сначала мысленно определи:

1. Есть ли в контексте информация для ответа:
   - полностью
   - частично
   - отсутствует
   - противоречива

2. Требует ли вопрос:
   - прямого ответа (да/нет)
   - сравнения (например: «так же», «аналогично»)
   - обобщения нескольких фактов
   - развёрнутого объяснения (например: «что такое», «какие», «как»)

---

## ПРАВИЛА ФОРМИРОВАНИЯ ОТВЕТА

### Общие правила:
- Не придумывай факты, причины, примеры или детали, которых нет в контексте.
- Не используй внешние знания.
- Не делай предположений без явной опоры на текст.
- Не добавляй рассуждений и объяснений процесса.
- Отвечай только по существу, без вводных фраз.

### Запрещённые выражения:
- «Ответ:»
- «В предоставленных документах…»
- «В контексте указано…»
- «Согласно источнику…»
- «В контексте сказано…»
- Любые другие ссылки на наличие контекста или документов.

### Работа с контекстом:
- Учитывай ВСЕ релевантные фрагменты, а не только первый.
- Если ответ содержится в нескольких частях контекста — ОБЯЗАТЕЛЬНО объедини их.
- Если есть и возможности, и ограничения — учитывай ОБА и делай итоговый вывод.

### Ограничения и приоритеты:
- Если в контексте есть прямые ограничения («не допускается», «не предусмотрено», «не существует») — они имеют приоритет.
- Не преобразовывай «возможно», «допускается», «может применяться» в однозначный ответ «да».
- Не обобщай свойства на всю группу, если они указаны только для части.

### Работа с вопросом:
- Строго учитывай формулировку вопроса.
- Не подменяй вопрос на более общий.

---

## КЛЮЧЕВОЕ ПРАВИЛО: ФОРМАТ ОТВЕТА

**Если вопрос допускает ответ «да» или «нет» (например: «Могу ли я…», «Положена ли…», «Обязан ли…»):**
- Начни ответ с «Да» или «Нет» (без кавычек).
- Затем добавь краткое пояснение, если необходимо.

**Если вопрос НЕ допускает ответа «да»/«нет» (например: «Что такое…», «Какие документы…», «Как подать…», «Какие платформы…»):**
- Запрещено начинать ответ с «Да» или «Нет».
- Ответ должен быть содержательным: описание, перечисление, объяснение.

**Примеры правильных ответов:**
- Вопрос: «Могу ли я получить стипендию?» → «Нет. Государственная социальная стипендия назначается только инвалидам I и II групп.»
- Вопрос: «Какие документы нужны?» → «Справка об инвалидности, копия ИПРА, заявление о создании специальных условий.»
- Вопрос: «Что такое особая квота?» → «Особая квота — это ежегодно устанавливаемый вузом объем бюджетных мест в размере не менее 10% от контрольных цифр приема.»

**Примеры неправильных ответов (запрещены):**
- «Да, согласно контексту…»
- «В предоставленных документах сказано, что…»
- «Ответ: Нет.»

---

## Работа с вопросом, содержащим сравнение (например: «так же», «аналогично»)
- Явно определи: совпадает ли формат полностью.
- Если нет — ответ должен быть «Нет» или «Частично», с пояснением.

## Работа с неполной информацией:
- Если информации достаточно для логического вывода — сделай вывод.
- Если есть только часть информации — ответь в её пределах и укажи ограничения.
- НЕ отвечай «нет информации», если можно сделать прямой вывод из контекста.

## Отсутствие информации:
- Если в контексте действительно нет данных для ответа → напиши: «Нет информации по этому вопросу.»

## Противоречия:
- Если в контексте есть противоречия — укажи это и не делай однозначного вывода.

---

## ВЫБОР ИЗ СПИСКА

Если в вопросе требуется определить «лучшие», «подходящие», «рекомендуемые»:

1. Найди все упомянутые варианты в контексте.
2. Найди ограничения и неподходящие варианты.
3. Исключи:
   - те, которые явно указаны как неподходящие
   - те, которые нарушают условия вопроса
4. Сформируй ответ:
   - перечисли подходящие варианты из контекста
   - не добавляй новых

**ВАЖНО:** Если есть список и есть критерий отбора → это ДОСТАТОЧНАЯ информация. Запрещено отвечать «нет информации», если можно сделать выбор из списка.

---

## СТИЛЬ ОТВЕТА

- Чётко, профессионально, без воды.
- Короткие абзацы или списки при необходимости.
- Без повторов.
- Без вводных фраз, ссылок на контекст и заголовка «Ответ:».

---

## КОНТЕКСТ

{context}

---

## ВОПРОС

{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    print(f"[STEP 5] Ответ от LLM получен за {time.time() - t0:.3f} сек")

    return response["message"]["content"]


def rag_pipeline(question):
    print("\n================ NEW QUERY ================")
    print(f"[QUESTION] {question}")

    total_start = time.time()

    results = retrieve(question)

    if not results or results[0].score < MIN_SIMILARITY_SCORE:
        print("[RESULT] Нет релевантных данных")
        return NO_ANSWER_MESSAGE, []

    print("[STEP 3] Реранкинг...")
    t2 = time.time()
    reranked_results = rerank(question, results, top_k=TOP_K_RERANK)
    print(f"[STEP 3] Реранкинг завершен за {time.time() - t2:.3f} сек")

    if not reranked_results:
        print("[RESULT] Реранкинг не дал результатов")
        return NO_ANSWER_MESSAGE, []

    context, sources = build_context(reranked_results)

    answer = generate(question, context)

    save_to_csv(question, results, reranked_results, answer)

    print(f"[DONE] Полный pipeline: {time.time() - total_start:.3f} сек")

    return answer, sources

# ================= UI =================
st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("🤖 RAG система для преподавателей")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            answer, sources = rag_pipeline(prompt)

            full_answer = answer

            if sources:
                full_answer += "\n\n**Источники:**\n"
                for s in set(sources):
                    full_answer += f"- {s}\n"

            st.markdown(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})