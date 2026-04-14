import streamlit as st
import os
import hashlib
import numpy as np
import json
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document
import tempfile
from scripts.chunking import chunk_pages
from scripts.create_embeddings import load_model, create_embeddings

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
@st.cache_resource
def get_model():
    return load_model()

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

def parse_file_to_pages(uploaded_file):
    pages = []
    file_name = uploaded_file.name.lower()

    # ===== TXT =====
    if file_name.endswith(".txt"):
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8")

        pages.append({
            "text": text,
            "page": 1,
            "source": uploaded_file.name
        })

    # ===== PDF =====
    elif file_name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)

        for i, page in enumerate(doc):
            text = page.get_text()

            pages.append({
                "text": text,
                "page": i + 1,
                "source": uploaded_file.name
            })

    # ===== DOCX =====
    elif file_name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        doc = Document(tmp_path)

        text = "\n".join([p.text for p in doc.paragraphs])

        pages.append({
            "text": text,
            "page": 1,
            "source": uploaded_file.name
        })

    else:
        return []

    return pages


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
        uploaded_files = st.file_uploader(
            "Выберите документы",
            type=["txt", "pdf", "docx", "doc"],
            accept_multiple_files=True
        )

        if uploaded_files:

            st.info(f"📂 Выбрано файлов: {len(uploaded_files)}")

            if st.button("📤 Загрузить в базу"):

                progress = st.progress(0)
                status = st.empty()

                all_pages = []

                # ===== 1. PARSE =====
                status.write("📄 Чтение файлов...")
                for i, uploaded_file in enumerate(uploaded_files):
                    pages = parse_file_to_pages(uploaded_file)
                    all_pages.extend(pages)

                    progress.progress((i + 1) / len(uploaded_files) * 0.2)

                # ===== 2. CHUNKING =====
                status.write("✂️ Разбиение на чанки...")
                chunks = chunk_pages(all_pages)
                progress.progress(0.4)

                if not chunks:
                    st.error("❌ Нет чанков")
                    st.stop()

                # ===== 3. EMBEDDINGS =====
                status.write("🧠 Создание embeddings...")

                model = get_model()

                embeddings = create_embeddings(
                    model,
                    chunks,
                    batch_size=32
                )

                progress.progress(0.8)

                # ===== 4. QDRANT =====
                status.write("📦 Загрузка в Qdrant...")

                points = []

                for chunk, vector in zip(chunks, embeddings):
                    points.append(
                        PointStruct(
                            id=hashlib.md5(chunk["chunk_id"].encode()).hexdigest(),
                            vector=vector.tolist(),
                            payload=chunk
                        )
                    )

                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    batch_size=128
                )

                progress.progress(1.0)

                # ===== DONE =====
                status.write("✅ Готово!")
                st.success(f"Загружено: {len(points)} чанков")