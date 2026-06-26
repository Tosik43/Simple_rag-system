import time

from core.retrieval import retrieve
from core.context import build_context
from core.generation_gigachat import generate
from core.config import MIN_SIMILARITY_SCORE, NO_ANSWER_MESSAGE, TOP_K_RERANK
from reranker import rerank
from core.logger import save_to_csv


def rag_pipeline(question, embed_model, qdrant):
    print("\n================ NEW QUERY ================")
    print(f"[QUESTION] {question}")

    total_start = time.time()

    # ===== RETRIEVE =====
    results, embed_time, search_time = retrieve(question, embed_model, qdrant)

    if not results or results[0].score < MIN_SIMILARITY_SCORE:
        print("[RESULT] Нет релевантных данных")
        return NO_ANSWER_MESSAGE, []

    # ===== RERANK =====
    print("[STEP 3] Реранкинг...")
    t2 = time.time()
    reranked_results = rerank(question, results, top_k=TOP_K_RERANK)
    rerank_time = time.time() - t2
    print(f"[STEP 3] Реранкинг завершен за {rerank_time:.3f} сек")

    if not reranked_results:
        print("[RESULT] Реранкинг не дал результатов")
        return NO_ANSWER_MESSAGE, []

    # ===== CONTEXT =====
    context, sources = build_context(reranked_results)

    # ===== GENERATE =====
    answer, gen_time = generate(question, context)

    # ===== LOG =====
    save_to_csv(
        question,
        results,
        reranked_results,
        answer,
        embed_time,
        search_time,
        rerank_time,
        gen_time
    )

    print(f"[DONE] Полный pipeline: {time.time() - total_start:.3f} сек")

    return answer, sources