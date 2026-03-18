from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")


def delete_pdf_by_source(source_name: str):
    client = QdrantClient(url=QDRANT_URL)

    print(f"Deleting document with source = '{source_name}'...")

    client.delete(
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

    print("✅ Document deleted successfully")


if __name__ == "__main__":
    delete_pdf_by_source("Сборник_ответов_на_наиболее_распространенные_вопросы_по_итогам_работы.pdf")