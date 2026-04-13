import streamlit as st
import os
import hashlib
import numpy as np
import json
from dotenv import load_dotenv

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue


from main import qdrant, COLLECTION_NAME, load_data

# ================= CONFIG =================
load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWD")

st.set_page_config(page_title="Admin Panel", layout="wide")

# ================= AUTH =================
if "auth" not in st.session_state:
    st.session_state.auth = False

# ================= LOGIN SCREEN =================
if not st.session_state.auth:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image("SUSU_logo.png", width=360)

        st.markdown("## 🔐 Вход в админ-панель")

        password = st.text_input("Пароль", type="password")

        if st.button("Войти"):
            if password == ADMIN_PASSWORD:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Неверный пароль")

    st.stop()

# ================= ADMIN PANEL =================
st.title("⚙️ Админ-панель")

# Кнопка выхода
if st.button("🚪 Выйти"):
    st.session_state.auth = False
    st.rerun()

st.divider()

# ================= FUNCTIONS =================
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


# ================= UI =================
st.subheader("📂 Управление документами")

docs = get_documents()

if not docs:
    st.warning("Нет загруженных документов")
else:
    selected_doc = st.selectbox("Документы", docs)

    col1, col2 = st.columns(2)

    # ================= DELETE =================
    with col1:
        if st.button("🗑 Удалить документ"):
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=selected_doc)
                        )
                    ]
                )
            )
            st.success(f"Удален: {selected_doc}")
            st.rerun()

    # ================= UPLOAD =================
    with col2:
        if st.button("⬆️ Загрузить документ"):
            embeddings, metadata = load_data()

            points = []

            for i in range(len(embeddings)):
                if metadata[i]["source"] == selected_doc:
                    points.append(
                        PointStruct(
                            id=hashlib.md5(metadata[i]["chunk_id"].encode()).hexdigest(),
                            vector=embeddings[i].tolist(),
                            payload=metadata[i]
                        )
                    )

            if points:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                st.success(f"Загружено: {len(points)} чанков")
            else:
                st.error("Документ не найден")