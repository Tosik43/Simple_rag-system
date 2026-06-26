import time

from core.config import COLLECTION_NAME, TOP_K_RETRIEVE

def retrieve(query: str, embed_model, qdrant):
    print("\n[STEP 1] Векторизация запроса...")
    t0 = time.time()
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    embed_time = time.time() - t0
    print(f"[STEP 1] Готово за {embed_time:.3f} сек")

    print("[STEP 2] Поиск в Qdrant...")
    t1 = time.time()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE
    )
    search_time = time.time() - t1
    print(f"[STEP 2] Найдено {len(results.points)} чанков за {search_time:.3f} сек")

    return results.points, embed_time, search_time