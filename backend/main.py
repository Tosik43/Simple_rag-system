from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
import hashlib
import time
from datetime import datetime
import fitz
from docx import Document

load_dotenv()

# Core imports
from core.pipeline import rag_pipeline
from core.init import load_models

# Scripts imports
from scripts.chunking import chunk_pages
from scripts.create_embeddings import load_model, create_embeddings

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

app = FastAPI(
    title="Simple RAG System API",
    description="FastAPI backend для RAG системы",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = None
QDRANT_CLIENT = None
COLLECTION_NAME = "test"

@app.on_event("startup")
async def startup_event():
    global EMBED_MODEL, QDRANT_CLIENT
    try:
        EMBED_MODEL, QDRANT_CLIENT = load_models()
        logger.info("✅ Models and Qdrant client loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        


# ====================== Схемы ======================
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranker: Optional[bool] = True

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict] = []
    query: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_count: int
    status: str
    processing_time: float


# ====================== Вспомогательная функция ======================
def parse_file_to_pages(file_path: str):
    pages = []
    filename = os.path.basename(file_path)

    try:
        # TXT
        if filename.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            pages.append({
                "text": content,
                "page": 1,
                "source": filename
            })

        # PDF
        elif filename.lower().endswith(".pdf"):
            doc = fitz.open(file_path)

            for i, page in enumerate(doc):
                text = page.get_text().strip()

                if text:
                    pages.append({
                        "text": text,
                        "page": i + 1,
                        "source": filename
                    })

            doc.close()

        # DOCX
        elif filename.lower().endswith(".docx"):
            doc = Document(file_path)

            text = "\n".join(
                p.text for p in doc.paragraphs
                if p.text.strip()
            ).strip()

            pages.append({
                "text": text,
                "page": 1,
                "source": filename
            })

        return pages

    except Exception as e:
        logger.exception(f"Ошибка парсинга {filename}: {e}")
        return []


# ====================== Эндпоинты ======================
@app.post("/query", response_model=RAGResponse)
async def query_rag(request: QueryRequest):
    if EMBED_MODEL is None or QDRANT_CLIENT is None:
        raise HTTPException(status_code=503, detail="Модели не загружены")
    
    try:
        answer, sources = rag_pipeline(request.query, EMBED_MODEL, QDRANT_CLIENT)
        return RAGResponse(
            answer=answer,
            sources=[{"content": str(s)} for s in sources],
            query=request.query
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Полная обработка: сохранение + парсинг + чанкинг + embeddings + Qdrant"""
    start_time = time.time()
    
    allowed = {".pdf", ".docx", ".txt"}
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail="Поддерживаются PDF, DOCX, TXT")

    try:
        # 1. Сохраняем файл
        upload_dir = "documents"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"Файл сохранён: {file_path}")
        logger.info(f"Размер файла: {os.path.getsize(file_path)} байт")

        pages = parse_file_to_pages(file_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="Не удалось извлечь текст"
            )

        # 3. Чанкинг
        chunks = chunk_pages(pages)
        if not chunks:
            raise HTTPException(status_code=400, detail="Не удалось создать чанки")

        # 4. Embeddings
        embeddings = create_embeddings(
            EMBED_MODEL,
            chunks,
            batch_size=64
        )

        # 5. Загрузка в Qdrant
        upload_time = datetime.utcnow().isoformat()
        points = [
            PointStruct(
                id=hashlib.md5(c["chunk_id"].encode()).hexdigest(),
                vector=v.tolist(),
                payload={**c, "upload_time": upload_time}
            )
            for c, v in zip(chunks, embeddings)
        ]

        QDRANT_CLIENT.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        processing_time = time.time() - start_time

        logger.info(f"✅ Успешно загружен: {file.filename} | Чанков: {len(chunks)} | Время: {processing_time:.2f}с")

        return UploadResponse(
            message="Документ успешно обработан и добавлен в базу знаний",
            filename=file.filename,
            chunks_count=len(chunks),
            status="success",
            processing_time=round(processing_time, 2)
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/documents")
async def list_documents():
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        return []
    return [f for f in os.listdir(docs_dir) if f.lower().endswith(('.pdf', '.docx', '.txt'))]


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Удаление файла + очистка чанков из Qdrant"""
    file_path = os.path.join("documents", filename)
    
    try:
        # 1. Удаляем файл с диска
        file_deleted = False
        if os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
            logger.info(f"Файл удалён с диска: {filename}")

        # 2. Удаляем чанки из Qdrant (правильный формат)
        try:
            delete_result = QDRANT_CLIENT.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=filename)
                        )
                    ]
                )
            )
            logger.info(f"✅ Чанки успешно удалены из Qdrant для {filename}")
        except Exception as qdrant_err:
            logger.warning(f"Не удалось удалить чанки из Qdrant: {qdrant_err}")

        return {
            "message": f"Документ '{filename}' успешно обработан",
            "file_deleted": file_deleted,
            "qdrant_cleaned": True
        }

    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": EMBED_MODEL is not None}