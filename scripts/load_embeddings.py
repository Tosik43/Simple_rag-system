import os
import json
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE")
METADATA_FILE = os.getenv("METADATA_FILE")
QDRANT_URL = os.getenv("QDRANT_URL")

VECTOR_SIZE = 1024


def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return embeddings, metadata

def create_collection(client: QdrantClient):
    print(f"Creating collection '{COLLECTION_NAME}'...")

    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting...")
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully")



def upload_embeddings(client, embeddings, metadata):
    points = []

    for i in range(len(embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload=metadata[i],
            )
        )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=128,
    )

    print("Embeddings uploaded")


def main():

    print("Loading data...")
    embeddings, metadata = load_data()

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    create_collection(client)

    print("Uploading embeddings...")
    upload_embeddings(client, embeddings, metadata)

    print("DONE")


if __name__ == "__main__":
    main()