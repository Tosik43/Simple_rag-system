from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from loguru import logger
import uvicorn


load_dotenv()

# Импорт существующего пайплайна
from core.pipeline import rag_pipeline
from core.init import load_models

app = FastAPI(
    title="Simple RAG System API",
    description="FastAPI backend для RAG системы",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замени на конкретные адреса
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
EMBED_MODEL = None
QDRANT_CLIENT = None

@app.on_event("startup")
async def startup_event():
    global EMBED_MODEL, QDRANT_CLIENT
    try:
        EMBED_MODEL, QDRANT_CLIENT = load_models()
        logger.info("✅ Models and Qdrant client loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")


# ====================== Pydantic схемы ======================
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranker: Optional[bool] = True


class Source(BaseModel):
    content: str
    metadata: Optional[dict] = {}


class RAGResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    query: str


# ====================== Эндпоинты ======================
@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    """Основной эндпоинт RAG"""
    if EMBED_MODEL is None or QDRANT_CLIENT is None:
        raise HTTPException(status_code=503, detail="Модели ещё не загружены. Подождите запуска.")

    try:
        answer, sources = rag_pipeline(
            request.query,
            EMBED_MODEL,
            QDRANT_CLIENT
        )

        # Преобразование sources
        source_list = [Source(content=str(s), metadata={}) for s in sources]

        return RAGResponse(
            answer=answer,
            sources=source_list,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": EMBED_MODEL is not None
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)