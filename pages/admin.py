import streamlit as st
import requests
import time
import tempfile
import os
from datetime import datetime
import fitz
from docx import Document

from core.config import ADMIN_PASSWORD, MAX_FILES, MAX_FILE_SIZE_MB

API_URL = "http://127.0.0.1:8000"

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

# ====================== ПРЕВЬЮ ======================
def preview_file(uploaded_file, page_num=0):
    name = uploaded_file.name.lower()
    preview = {"type": "text", "data": "Не удалось показать предпросмотр", "total_pages": 1}

    try:
        uploaded_file.seek(0)
        contents = uploaded_file.read()

        if name.endswith(".pdf"):
            doc = fitz.open(stream=contents, filetype="pdf")
            total_pages = len(doc)
            page_num = max(0, min(page_num, total_pages - 1))
            
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("png")
            
            doc.close()

            preview = {
                "type": "image",
                "data": img_bytes,
                "total_pages": total_pages,
                "current_page": page_num + 1
            }

        elif name.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(contents)
                path = tmp.name
            doc = Document(path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            os.remove(path)
            preview = {"type": "text", "data": text[:1500] + ("..." if len(text) > 1500 else ""), "total_pages": 1}

        elif name.endswith(".txt"):
            text = contents.decode("utf-8")
            words = text.split()[:100]
            preview_text = " ".join(words) + ("..." if len(words) == 100 else "")
            preview = {"type": "text", "data": preview_text, "total_pages": 1}

    except Exception as e:
        preview["data"] = f"Ошибка предпросмотра: {e}"

    uploaded_file.seek(0)
    return preview


# ================= DOCUMENTS =================
st.subheader("📂 Управление документами")

col1, col2 = st.columns(2)

# DELETE
with col1:
    try:
        docs_response = requests.get(f"{API_URL}/documents")
        docs = docs_response.json() if docs_response.status_code == 200 else []
        st.caption(f"📊 Документов в системе: {len(docs)}")
        
        if docs:
            selected_doc = st.selectbox("Документы", docs)
            
            if st.button("Удалить документ", type="primary"):
                try:
                    r = requests.delete(f"{API_URL}/documents/{selected_doc}")
                    if r.status_code == 200:
                        st.success(f"✅ Удален: {selected_doc}")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"Ошибка: {r.text}")
                except Exception as e:
                    st.error(f"Ошибка соединения: {e}")
        else:
            st.warning("Нет документов")
    except:
        st.error("Не удалось подключиться к FastAPI")

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

        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"{f.name} превышает {MAX_FILE_SIZE_MB} MB")
                st.stop()

        st.info(f"📂 Файлов к загрузке: {len(uploaded_files)}")

        # Превью
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                if file.name.lower().endswith(".pdf"):
                    # Листание страниц
                    key = f"pdf_page_{file.name}"
                    if key not in st.session_state:
                        st.session_state[key] = 0

                    col_prev, col_info, col_next = st.columns([1, 3, 1])
                    with col_prev:
                        if st.button("←", key=f"prev_{file.name}"):
                            st.session_state[key] = max(0, st.session_state[key] - 1)
                    with col_info:
                        st.write(f"Страница **{st.session_state[key] + 1}**")
                    with col_next:
                        if st.button("→", key=f"next_{file.name}"):
                            st.session_state[key] += 1

                    preview = preview_file(file, st.session_state[key])
                    if preview["type"] == "image":
                        st.image(preview["data"])
                else:
                    preview = preview_file(file, 0)
                    st.text(preview["data"])

        if st.button("📤 Загрузить в базу"):
            progress = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded_files):
                status.write(f"Загрузка: {file.name} ({i+1}/{len(uploaded_files)})")
                
                try:
                    files_data = {"file": (file.name, file.getvalue(), file.type)}
                    response = requests.post(f"{API_URL}/upload", files=files_data)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ {file.name} — {data.get('chunks_count', 0)} чанков")
                    else:
                        st.error(f"❌ {file.name}: {response.text}")
                except Exception as e:
                    st.error(f"Ошибка загрузки {file.name}: {e}")
                
                progress.progress((i + 1) / len(uploaded_files))

            st.success("✅ Документы успешно загружены!")
            time.sleep(2)
            st.rerun()
