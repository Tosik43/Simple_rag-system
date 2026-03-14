import os
import ollama

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from reranker import rerank
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

TOP_K_RETRIEVE = 30
TOP_K_RERANK = 5

MIN_SIMILARITY_SCORE = 0.6

NO_ANSWER_MESSAGE = (
    "В предоставленных документах нет информации по этому вопросу. "
    "Попробуйте задать ваш вопрос по другому."
)

app = FastAPI(title="RAG API")

print("Loading embedding model...")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL)

print("API READY")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


def retrieve(query: str):
    query_vector = EMBED_MODEL.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE
    )

    return results.points


def build_context(results):
    context = ""
    sources = []

    for i, r in enumerate(results):
        text = r["text"]
        source = r["source"]
        page = r["page"]

        context += f"""
Источник {i+1}
Документ: {source}
Страница: {page}
Текст:
{text}
"""

        sources.append(f"{source} (стр. {page})")

    return context, sources


def generate(query, context):
    prompt = f"""
Ты — AI ассистент университета.

Отвечай ТОЛЬКО на основе предоставленной информации.

Если ответа в документах нет, напиши:
В предоставленных документах нет информации по этому вопросу.

Правила:
- не придумывай информацию
- отвечай ясно и профессионально
- не объясняй свою работу

Информация:
{context}

Вопрос:
{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0.2
        }
    )

    return response["message"]["content"]


def rag_pipeline(question):

    results = retrieve(question)

    if not results:
        return NO_ANSWER_MESSAGE

    # Проверка релевантности поиска
    if results[0].score < MIN_SIMILARITY_SCORE:
        return NO_ANSWER_MESSAGE

    reranked_results = rerank(
        question,
        results,
        top_k=TOP_K_RERANK
    )

    if not reranked_results:
        return NO_ANSWER_MESSAGE

    context, sources = build_context(reranked_results)

    answer = generate(question, context)

    answer_lower = answer.lower()

    # Проверка ответа модели
    if "нет информации" in answer_lower:
        return NO_ANSWER_MESSAGE

    # Добавляем источники только если ответ найден
    answer_with_sources = answer + "\n\nИсточники:\n"

    for source in set(sources):
        answer_with_sources += f"- {source}\n"

    return answer_with_sources


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    answer = rag_pipeline(req.question)
    return AnswerResponse(answer=answer)


@app.get("/")
def health():
    return {"status": "ok"}