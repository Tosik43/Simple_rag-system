import streamlit as st
import os
import hashlib
import fitz
from docx import Document
import tempfile
import time

from datetime import datetime
from core.config import *

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from scripts.chunking import chunk_pages
from scripts.create_embeddings import load_model, create_embeddings


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

# ================= CHECK QDRANT =================
if not qdrant:
    st.error("❌ Qdrant не настроен")
    st.stop()

# ================= UI =================
st.title("⚙️ Админ-панель")

if st.button("🚪 Выйти"):
    st.session_state.auth = False
    st.rerun()

st.divider()

# ================= FUNCTIONS =================
def format_date(dt_str):
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%d.%m.%Y %H:%M")
    except:
        return "—"


@st.cache_resource
def get_model():
    return load_model()

@st.cache_data(ttl=60)
def get_documents():
    sources = {}
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
            upload_time = point.payload.get("upload_time")

            if source and source not in sources:
                sources[source] = upload_time

        if next_page is None:
            break

        offset = next_page

    return sources


def parse_file_to_pages(uploaded_file):
    pages = []
    name = uploaded_file.name.lower()

    try:
        # TXT
        if name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8").strip()
            pages.append({"text": text, "page": 1, "source": uploaded_file.name})

        # PDF
        elif name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            doc = fitz.open(path)

            for i, page in enumerate(doc):
                pages.append({
                    "text": page.get_text().strip(),
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
            text = "\n".join([p.text for p in doc.paragraphs]).strip()

            os.remove(path)

            pages.append({
                "text": text,
                "page": 1,
                "source": uploaded_file.name
            })

    except Exception as e:
        st.error(f"Ошибка при чтении файла {uploaded_file.name}: {e}")

    return pages


def preview_file(uploaded_file):
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            doc = fitz.open(path)

            img_bytes = None

            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")

            doc.close()
            os.remove(path)

            uploaded_file.seek(0)

            if img_bytes:
                return {"type": "image", "data": img_bytes}

        elif name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")

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

    except Exception as e:
        return {"type": "text", "data": f"Ошибка предпросмотра: {e}"}


def safe_upsert(points, progress, start_progress=0.8):
    BATCH_SIZE = 128

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]

        for _ in range(3):
            try:
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                break
            except Exception as e:
                st.warning(f"Ошибка загрузки батча: {e}")
                time.sleep(2)

        progress.progress(
            min(1.0, start_progress + (i + len(batch)) / len(points) * 0.2)
        )


# ================= DOCUMENTS =================
st.subheader("📂 Управление документами")

docs = get_documents()
st.caption(f"📊 В базе документов: {len(docs)}")

col1, col2 = st.columns(2)

# DELETE
with col1:
    if docs:
        selected_doc = st.selectbox(
            "Документы",
            list(docs.keys()),
            format_func=lambda x: f"{x} (🕒 {format_date(docs[x])})"
        )

        if st.button("🗑 Удалить документ"):
            try:
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
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка удаления: {e}")
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

        if len(uploaded_files) > MAX_FILES:
            st.error(f"Максимум {MAX_FILES} файлов за раз")
            st.stop()


        # 🔒 проверка размера
        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"{f.name} превышает {MAX_FILE_SIZE_MB} MB")
                st.stop()

        st.info(f"📂 Файлов к загрузке: {len(uploaded_files)}")

        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                preview = preview_file(file)
                if preview["type"] == "image":
                    st.image(preview["data"])
                else:
                    st.code(preview["data"])

        existing_docs = {d.lower() for d in docs.keys()}

        duplicate_files = [
            f.name for f in uploaded_files
            if f.name.lower() in existing_docs
        ]

        if duplicate_files:
            st.error("❌ Уже есть:\n" + "\n".join(duplicate_files))
            st.stop()

        if st.button("📤 Загрузить в базу"):

            try:
                progress = st.progress(0)
                status = st.empty()

                # PARSE
                status.write("📄 Чтение файлов...")
                all_pages = []

                for i, file in enumerate(uploaded_files):
                    pages = parse_file_to_pages(file)
                    all_pages.extend(pages)
                    progress.progress((i + 1) / len(uploaded_files) * 0.2)

                # CHUNK
                status.write("✂️ Разбиение...")
                chunks = chunk_pages(all_pages)
                progress.progress(0.4)

                if not chunks:
                    st.error("Нет чанков")
                    st.stop()

                # EMBEDDINGS
                status.write("🧠 Embeddings...")
                model = get_model()
                embeddings = create_embeddings(model, chunks, batch_size=64)
                progress.progress(0.8)

                # QDRANT
                status.write("📦 Загрузка...")
                upload_time = datetime.utcnow().isoformat()

                points = [
                    PointStruct(
                        id=hashlib.md5(c["chunk_id"].encode()).hexdigest(),
                        vector=v.tolist(),
                        payload={**c, "upload_time": upload_time}
                    )
                    for c, v in zip(chunks, embeddings)
                ]

                safe_upsert(points, progress)

                progress.progress(1.0)

                # DONE
                status.write("✅ Готово!")
                st.success("✅ Документы успешно загружены!")
                st.info(f"Добавлено чанков: {len(points)}")

                st.cache_data.clear()

                with st.spinner("⏳ Обновление страницы..."):
                    time.sleep(5)

                st.rerun()

            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {e}")