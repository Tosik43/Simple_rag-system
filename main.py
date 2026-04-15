import streamlit as st

from core.init import load_models
from core.pipeline import rag_pipeline

# ================= INIT =================
st.set_page_config(page_title="RAG Chat", layout="wide")

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

            answer, sources = rag_pipeline(prompt,
                EMBED_MODEL,
                qdrant
            )

            full_answer = answer

            if sources:
                full_answer += "\n\n**Источники:**\n"
                for s in set(sources):
                    full_answer += f"- {s}\n"

            st.markdown(full_answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer
    })