import json

def merge_ground_truth(source_file, target_file, output_file=None):
    """
    Копирует поле ground_truth из source_file в target_file.
    
    Args:
        source_file (str): путь к файлу, содержащему ground_truth
        target_file (str): путь к файлу, в который нужно добавить ground_truth
        output_file (str, optional): путь к выходному файлу. Если не указан, 
                                     используется target_file (файл будет перезаписан)
    """
    # Загружаем данные из первого файла (целевой)
    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    # Загружаем данные из второго файла (источник ground_truth)
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Проверяем, что количество элементов совпадает
    if len(target_data) != len(source_data):
        print(f"Предупреждение: количество записей в файлах различается:")
        print(f"  {target_file}: {len(target_data)} записей")
        print(f"  {source_file}: {len(source_data)} записей")
        
        # Запрашиваем подтверждение у пользователя
        response = input("Продолжить копирование только для существующих индексов? (y/n): ")
        if response.lower() != 'y':
            print("Операция отменена.")
            return
    
    # Копируем ground_truth из source в target
    min_len = min(len(target_data), len(source_data))
    copied_count = 0
    
    for i in range(min_len):
        if 'ground_truth' in source_data[i]:
            target_data[i]['ground_truth'] = source_data[i]['ground_truth']
            copied_count += 1
        else:
            print(f"Предупреждение: в записи {i} исходного файла отсутствует поле 'ground_truth'")
    
    # Определяем выходной файл
    if output_file is None:
        output_file = target_file
    
    # Сохраняем результат
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! Скопировано {copied_count} полей 'ground_truth'.")
    print(f"Результат сохранён в: {output_file}")


if __name__ == "__main__":
    # Пример использования
    # Укажите пути к вашим файлам
    first_file = "evaluation_results/ragas_results_20260329_123601.json"      # файл с пустыми ground_truth
    second_file = "evaluation_results/ragas_results_20260329_123601_ground_truth.json"            # файл с заполненными ground_truth
    
    # Выполнить слияние (перезапишет первый файл - ВНИМАНИЕ!)
    # merge_ground_truth(second_file, first_file)
    
    # Или сохранить в новый файл
    merge_ground_truth(second_file, first_file, "ragas_results_merged.json")