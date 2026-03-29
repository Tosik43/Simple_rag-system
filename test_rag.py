import json
import requests
import time
from typing import List, Dict, Any
from tqdm import tqdm

# НАСТРОЙКИ - УВЕЛИЧЕН ТАЙМАУТ
API_URL = "http://127.0.0.1:8000/ask"
MAX_RETRIES = 2  # Уменьшим количество попыток, но увеличим таймаут
RETRY_DELAY = 2  # Увеличен delay между попытками
TIMEOUT = 3600  # Увеличен таймаут до 120 секунд (2 минуты)

def load_questions(file_path: str) -> List[Dict[str, Any]]:
    """Загружает вопросы из JSON файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_answer_with_retry(question: str, max_retries: int = MAX_RETRIES) -> str:
    """Отправляет вопрос с повторными попытками при ошибках"""
    for attempt in range(max_retries):
        try:
            print(f"    Отправка запроса (попытка {attempt + 1}/{max_retries})...")
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()["answer"]
            else:
                error_msg = f"API Error (HTTP {response.status_code}): {response.text[:200]}"
                if attempt < max_retries - 1:
                    print(f"    Ошибка, повтор через {RETRY_DELAY} сек...")
                    time.sleep(RETRY_DELAY)
                    continue
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout error (превышено {TIMEOUT} сек)"
            if attempt < max_retries - 1:
                print(f"    Таймаут, повтор через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
                continue
            return error_msg
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            if attempt < max_retries - 1:
                print(f"    Ошибка подключения, повтор через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
                continue
            return error_msg
            
        except Exception as e:
            error_msg = f"Unknown error: {str(e)}"
            if attempt < max_retries - 1:
                print(f"    Ошибка, повтор через {RETRY_DELAY} сек...")
                time.sleep(RETRY_DELAY)
                continue
            return error_msg
    
    return "Max retries exceeded"

def test_api_connection():
    """Тестирует соединение с API"""
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=10)
        if response.status_code == 200:
            print("✅ API доступен")
            return True
        else:
            print(f"❌ API вернул статус: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Не удалось подключиться к API: {e}")
        print(f"   Убедитесь, что API запущен на http://127.0.0.1:8000")
        return False

def process_questions(input_file: str, output_file: str, start_from: int = 0, limit: int = None):
    """Обрабатывает вопросы с возможностью продолжить с определенного места"""
    
    print(f"Загрузка вопросов из {input_file}...")
    questions_data = load_questions(input_file)
    
    # Если нужно начать с определенного вопроса
    if start_from > 0:
        questions_data = questions_data[start_from:]
        print(f"Продолжаем с вопроса {start_from + 1}")
    
    # Если нужно ограничить количество вопросов
    if limit:
        questions_data = questions_data[:limit]
        print(f"Ограничение: {limit} вопросов")
    
    total = len(questions_data)
    print(f"Будет обработано {total} вопросов")
    print("-" * 50)
    
    # Тестируем соединение с API
    if not test_api_connection():
        print("\n❌ Невозможно продолжить - API не отвечает")
        return
    
    results = []
    
    # Используем tqdm для отображения прогресса
    for idx, item in enumerate(tqdm(questions_data, desc="Обработка вопросов", unit="вопрос"), start_from + 1):
        question = item.get("question", "")
        
        print(f"\n[{idx}/{total + start_from}] {question[:80]}...")
        
        # Получаем ответ от RAG системы
        start_time = time.time()
        answer = get_answer_with_retry(question)
        elapsed_time = time.time() - start_time
        
        print(f"    Время ответа: {elapsed_time:.1f} сек")
        answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
        print(f"    Ответ: {answer_preview}")
        
        # Сохраняем исходные поля + ответ
        result = {
            "question": question,
            "contexts": item.get("contexts", []),
            "answer": answer,
            "ground_truth": item.get("ground_truth", "")
        }
        
        results.append(result)
        
        # Сохраняем промежуточные результаты каждые 10 вопросов
        if idx % 10 == 0:
            temp_file = output_file.replace('.json', f'_temp_{idx}.json')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"    💾 Промежуточное сохранение: {temp_file}")
        
        # Небольшая задержка между запросами
        time.sleep(0.5)
    
    # Сохраняем финальные результаты
    print(f"\n💾 Сохранение результатов в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Статистика
    total_processed = len(results)
    errors = sum(1 for r in results if "Ошибка" in r["answer"] or 
                 "error" in r["answer"].lower() or 
                 "Timeout" in r["answer"] or
                 "Connection" in r["answer"])
    
    print("\n" + "=" * 60)
    print("СТАТИСТИКА:")
    print(f"  Всего вопросов: {total_processed}")
    print(f"  Успешно обработано: {total_processed - errors}")
    print(f"  С ошибками: {errors}")
    if errors > 0:
        print("\n  Вопросы с ошибками:")
        for i, r in enumerate(results):
            if "Ошибка" in r["answer"] or "error" in r["answer"].lower():
                print(f"    {i+1}. {r['question'][:60]}...")
                print(f"       Ошибка: {r['answer'][:100]}")
    print(f"\n  Результаты сохранены в: {output_file}")
    print("=" * 60)

def main():
    input_file = "evaluation_results/ragas_dataset_20260329_123600.json"
    output_file = "evaluation_results/ragas_results_20260329_123601.json"
    
    # Можно указать, с какого вопроса продолжить (если был сбой)
    START_FROM = 0  # Измените на номер вопроса, с которого нужно продолжить
    LIMIT = 169      # Установите для тестирования небольшого количества вопросов
    
    print("=" * 60)
    print("RAG ТЕСТИРОВАНИЕ")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Таймаут: {TIMEOUT} секунд")
    print(f"Входной файл: {input_file}")
    print(f"Выходной файл: {output_file}")
    if START_FROM > 0:
        print(f"Начинаем с вопроса: {START_FROM + 1}")
    if LIMIT:
        print(f"Лимит вопросов: {LIMIT}")
    print("=" * 60)
    print()
    
    process_questions(input_file, output_file, start_from=START_FROM, limit=LIMIT)

if __name__ == "__main__":
    main()