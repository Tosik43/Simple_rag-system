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


app = FastAPI(title="RAG API")

print("Loading embedding model...")
EMBED_MODEL_NAME = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL)

print("API READY")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


def retrieve(query: str):
    query_vector = EMBED_MODEL_NAME.encode(
        query,
        normalize_embeddings=True
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
        context += f"""
Источник {i+1}
Документ: {r['source']}
Страница: {r['page']}
Текст:
{r['text']}
"""
        sources.append(f"{r['source']} (стр. {r['page']})")

    return context, sources


def generate(query, context):
    prompt = f"""
Ты — AI ассистент университета.

Отвечай ТОЛЬКО на основе предоставленного контекста.

Правила:
- если ответа нет в контексте — скажи: "В предоставленных документах нет информации по этому вопросу"
- не придумывай
- отвечай ясно и профессионально
- не упоминай слово "контекст"
- не объясняй свою работу

Контекст:
{context}

Вопрос:
{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def rag_pipeline(question):
    
    results = retrieve(question)
    
    reranked_results = rerank(question, results, top_k=TOP_K_RERANK)
    
    context, sources = build_context(reranked_results)
    
    answer = generate(question, context)
    
    answer_with_sources = f"{answer}\n\nИсточники:\n"
    for source in sources:
        answer_with_sources += f"- {source}\n"

    return answer_with_sources


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    answer = rag_pipeline(req.question)
    return AnswerResponse(answer=answer)


@app.get("/")
def health():
    return {"status": "ok"}