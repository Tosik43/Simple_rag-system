import os
import json
import hashlib
import numpy as np
import ollama

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from dotenv import load_dotenv
from reranker import rerank

load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE")
METADATA_FILE = os.getenv("METADATA_FILE")

TOP_K_RETRIEVE = 30
TOP_K_RERANK = 3
MIN_SIMILARITY_SCORE = 0.1

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


def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata


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
        text = r.payload["text"]
        source = r.payload["source"]
        page = r.payload["page"]

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
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    return response["message"]["content"]


def rag_pipeline(question):
    results = retrieve(question)

    # if not results or results[0].score < MIN_SIMILARITY_SCORE:
    #     return NO_ANSWER_MESSAGE

    reranked_results = rerank(question, results)

    # if not reranked_results:
    #     return NO_ANSWER_MESSAGE
    
    reranked_results = reranked_results[:TOP_K_RERANK]

    context, sources = build_context(reranked_results)

    answer = generate(question, context)

    # if "нет информации" in answer.lower():
    #     return NO_ANSWER_MESSAGE

    answer += "\n\nИсточники:\n"
    for source in set(sources):
        answer += f"- {source}\n"

    return answer


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    return AnswerResponse(answer=rag_pipeline(req.question))


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/documents")
def get_documents():
    sources = set()
    offset = None

    while True:
        points, next_page = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=100,
            offset=offset
        )

        if not points:
            break

        for point in points:
            source = point.payload.get("source")
            if source:
                sources.add(source)

        if next_page is None:
            break

        offset = next_page

    return sorted(list(sources))


@app.delete("/documents/{source_name}")
def delete_document(source_name: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source_name)
                )
            ]
        )
    )

    return {"status": "deleted", "document": source_name}


@app.post("/documents/upload/{source_name}")
def upload_document(source_name: str):
    embeddings, metadata = load_data()

    points = []

    for i in range(len(embeddings)):
        if metadata[i]["source"] == source_name:
            points.append(
                PointStruct(
                    id=hashlib.md5(
                        metadata[i]["chunk_id"].encode("utf-8")
                    ).hexdigest(),
                    vector=embeddings[i].tolist(),
                    payload=metadata[i],
                )
            )

    if not points:
        return {"status": "error", "message": "document not found"}

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {"status": "uploaded", "chunks": len(points)}

# ================= ADMIN UI =================

@app.get("/admin", response_class=HTMLResponse)
def admin_panel():
    docs = get_documents()

    options = "".join(
        [f'<option value="{doc}">{doc}</option>' for doc in docs]
    )

    return f"""
    <html>
    <body>
        <h2>Удаление документа</h2>

        <select id="docSelect">
            {options}
        </select>

        <button onclick="deleteDoc()">Удалить</button>

        <script>
            async function deleteDoc() {{
                const doc = document.getElementById("docSelect").value;

                await fetch(`/documents/${{doc}}`, {{
                    method: "DELETE"
                }});

                alert("Удалено: " + doc);
                location.reload();
            }}
        </script>
    </body>
    </html>
    """