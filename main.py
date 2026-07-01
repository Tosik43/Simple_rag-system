import streamlit as st
import requests

from core.init import load_models
from core.pipeline import rag_pipeline

# ================= INIT =================
st.set_page_config(page_title="RAG Chat", layout="wide")

API_URL = "http://127.0.0.1:8000"

EMBED_MODEL, qdrant = load_models()

# ================= UI =================
st.title("🤖 RAG система для преподавателей")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===== История чата =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ===== Ввод пользователя =====
if prompt := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # ===== Ответ модели =====
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": prompt,
                        "top_k": 5,
                        "use_reranker": True
                    },
                    timeout=500
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Не удалось получить ответ")
                    sources = data.get("sources", [])

                    full_answer = answer

                    if sources:
                        full_answer += "\n\n**Источники:**\n"
                        for s in sources:
                            content = s.get("content", str(s))
                            full_answer += f"- {content[:300]}...\n"

                    st.markdown(full_answer)

                else:
                    st.error(f"Ошибка сервера: {response.status_code} - {response.text}")
                    full_answer = "Произошла ошибка при обработке запроса."

            except requests.exceptions.ConnectionError:
                st.error("❌ Не удалось подключиться к FastAPI серверу. Убедитесь, что он запущен.")
                full_answer = "Ошибка подключения к серверу."
            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
                full_answer = "Произошла непредвиденная ошибка."

    # Добавляем ответ ассистента в историю
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer
    })