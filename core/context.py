import time

def build_context(results):
    print("[STEP 4] Сбор контекста...")
    t0 = time.time()

    context = ""
    sources = []

    for i, r in enumerate(results):
        text = r.payload["text"]
        source = r.payload["source"]
        page = r.payload["page"]

        context += f"""
Источник {i+1}
Документ: {source}
Страница: {page}
{text}
"""

        sources.append(f"{source} (стр. {page})")

    print(f"[STEP 4] Контекст собран за {time.time() - t0:.3f} сек")

    return context, sources