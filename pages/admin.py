import streamlit as st
import requests
import time
from datetime import datetime

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

# ================= FUNCTIONS =================
def format_date(dt_str):
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%d.%m.%Y %H:%M")
    except:
        return "—"


# ================= DOCUMENTS =================
st.subheader("📂 Управление документами")

# Получаем список документов через API
try:
    docs_response = requests.get(f"{API_URL}/documents")
    docs = docs_response.json() if docs_response.status_code == 200 else []
except:
    docs = []
    st.error("Не удалось подключиться к FastAPI")

st.caption(f"📊 В базе документов: {len(docs)}")

col1, col2 = st.columns(2)

# DELETE
with col1:
    if docs:
        selected_doc = st.selectbox(
            "Документы",
            docs,
            format_func=lambda x: f"{x}"
        )

        if st.button("🗑 Удалить документ"):
            try:
                r = requests.delete(f"{API_URL}/documents/{selected_doc}")
                if r.status_code == 200:
                    st.success(f"Удален: {selected_doc}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Ошибка удаления: {r.text}")
            except Exception as e:
                st.error(f"Ошибка соединения: {e}")
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

        # Проверка размера
        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.error(f"{f.name} превышает {MAX_FILE_SIZE_MB} MB")
                st.stop()

        st.info(f"📂 Файлов к загрузке: {len(uploaded_files)}")

        # Предпросмотр оставляем как был
        for file in uploaded_files:
            with st.expander(f"📄 {file.name}"):
                # Можно оставить preview_file, если хочешь
                st.info(f"Размер: {round(file.size/1024, 1)} KB")

        if st.button("📤 Загрузить в базу"):

            try:
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

            except Exception as e:
                st.error(f"❌ Ошибка загрузки: {e}")