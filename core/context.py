import time

def build_context(results):
    print("[STEP 4] Сбор контекста...")
    t0 = time.time()

    context = ""
    sources = []

    for i, r in enumerate(results):
        text = r.payload.get("text")
        source = r.payload.get("source")
        page = r.payload.get("page")

        context += f"""
Источник {i+1}
Документ: {source}
Страница: {page}
{text}
"""

        # Кликабельная ссылка
        pdf_url = f"/documents/{source}#page={page}"
        sources.append(f"[{source} (стр. {page})]({pdf_url})")

    print(f"[STEP 4] Контекст собран за {time.time() - t0:.3f} сек")

    return context, sources