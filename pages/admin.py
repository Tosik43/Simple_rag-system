import streamlit as st
import os
import hashlib
from dotenv import load_dotenv
import fitz
from docx import Document
import tempfile
import time

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from scripts.chunking import chunk_pages
from scripts.create_embeddings import load_model, create_embeddings

# ================= CONFIG =================
load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWD")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

qdrant = QDRANT_URL and QdrantClient(url=QDRANT_URL)

st.set_page_config(page_title="Admin Panel", layout="wide")

# ================= AUTH =================
if "auth" not in st.session_state:
    st.session_state.auth = False

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

# ================= UI =================
st.title("⚙️ Админ-панель")

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
    name = uploaded_file.name.lower()

    # TXT
    if name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
        pages.append({"text": text, "page": 1, "source": uploaded_file.name})

    # PDF
    elif name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        doc = fitz.open(path)

        for i, page in enumerate(doc):
            pages.append({
                "text": page.get_text(),
                "page": i + 1,
                "source": uploaded_file.name
            })

        doc.close()
        os.remove(path)

    # DOCX
    elif name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])

        os.remove(path)

        pages.append({
            "text": text,
            "page": 1,
            "source": uploaded_file.name
        })

    return pages

def preview_file(uploaded_file):
    name = uploaded_file.name.lower()

    # ===== PDF → IMAGE =====
    if name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        doc = fitz.open(path)

        img_bytes = None

        if len(doc) > 0:
            page = doc[0]
            matrix = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=matrix)
            img_bytes = pix.tobytes("png")

        doc.close()
        os.remove(path)

        uploaded_file.seek(0)

        if img_bytes:
            return {"type": "image", "data": img_bytes}

    # ===== TXT =====
    elif name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    # ===== DOCX =====
    elif name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])

        os.remove(path)

    else:
        text = "Не удалось показать предпросмотр"

    uploaded_file.seek(0)

    return {"type": "text", "data": text[:2000]}

def safe_upsert(points, progress, start_progress=0.8):
    BATCH_SIZE = 128

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        for _ in range(3):  # retry
            try:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                break
            except Exception as e:
                print(f"[QDRANT ERROR] {e}")
                time.sleep(2)

        progress.progress(
            min(1.0, start_progress + (i + len(batch)) / len(points) * 0.2)
        )


# ================= DOCUMENTS =================
st.subheader("📂 Управление документами")

docs = get_documents()

col1, col2 = st.columns(2)

# DELETE
with col1:
    if docs:
        selected_doc = st.selectbox("Документы", docs)

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
    else:
        st.warning("Нет документов")

# ================= UPLOAD =================
with col2:
    uploaded_files = st.file_uploader(
        "Загрузить документы",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.info(f"📂 Выбрано файлов: {len(uploaded_files)}")

        st.markdown("### 👀 Предпросмотр файлов")

        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}"):
                preview = preview_file(file)

                if preview["type"] == "image":
                    st.image(preview["data"], caption="Первая страница PDF")

                else:
                    st.code(preview["data"], language="text")

        existing_docs = set(get_documents())

        duplicate_files = [
            f.name for f in uploaded_files
            if f.name in existing_docs
        ]

        if duplicate_files:
            st.error(
                "❌ Такие документы уже есть в базе:\n\n" +
                "\n".join(duplicate_files)
            )
            st.stop()

        if st.button("📤 Загрузить в базу"):

            progress = st.progress(0)
            status = st.empty()

            # ===== 1. PARSE =====
            status.write("📄 Чтение файлов...")
            all_pages = []

            for i, file in enumerate(uploaded_files):
                pages = parse_file_to_pages(file)
                all_pages.extend(pages)
                progress.progress((i + 1) / len(uploaded_files) * 0.2)

            # ===== 2. CHUNKING =====
            status.write("✂️ Разбиение на чанки...")
            chunks = chunk_pages(all_pages)
            progress.progress(0.4)

            if not chunks:
                st.error("Нет чанков")
                st.stop()

            # ===== 3. EMBEDDINGS =====
            status.write("🧠 Создание embeddings...")
            model = get_model()

            embeddings = create_embeddings(
                model,
                chunks,
                batch_size=64
            )

            progress.progress(0.8)

            # ===== 4. QDRANT =====
            status.write("📦 Загрузка в Qdrant...")

            points = [
                PointStruct(
                    id=hashlib.md5(c["chunk_id"].encode()).hexdigest(),
                    vector=v.tolist(),
                    payload=c
                )
                for c, v in zip(chunks, embeddings)
            ]

            safe_upsert(points, progress)

            progress.progress(1.0)

            # ===== DONE =====
            status.write("✅ Готово!")
            st.success(f"Загружено: {len(points)} чанков")

            with st.spinner("⏳ Обновление страницы..."):
                time.sleep(5)

            st.rerun()