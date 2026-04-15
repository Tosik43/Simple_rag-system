import time
import streamlit as st

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from core.config import EMBED_MODEL_NAME, QDRANT_URL

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