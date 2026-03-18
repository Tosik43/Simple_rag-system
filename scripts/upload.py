import os
import json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE")
METADATA_FILE = os.getenv("METADATA_FILE")


def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata


def upload_pdf_by_source(source_name: str):
    print(f"📄 Upload document: {source_name}")

    client = QdrantClient(url=QDRANT_URL)

    embeddings, metadata = load_data()

    points = []

    for i in range(len(embeddings)):
        if metadata[i]["source"] == source_name:
            points.append(
                PointStruct(
                    id=i,  # исправить надо
                    vector=embeddings[i].tolist(),
                    payload=metadata[i],
                )
            )

    print(f"📦 Найдено {len(points)} чанков")

    if not points:
        print("⚠️ Документ не найден, загрузка отменена")
        return

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=128,
    )

    print("✅ Документ успешно загружен")


if __name__ == "__main__":
    upload_pdf_by_source("Сборник_ответов_на_наиболее_распространенные_вопросы_по_итогам_работы.pdf")