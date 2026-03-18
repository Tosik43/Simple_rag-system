import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")


def get_documents():
    client = QdrantClient(url=QDRANT_URL)

    sources = set()
    offset = None

    while True:
        points, next_page = client.scroll(
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

    documents = sorted(list(sources))

    return documents


if __name__ == "__main__":
    docs = get_documents()

    print("📄 Документы в базе:")
    for doc in docs:
        print(f"- {doc}")