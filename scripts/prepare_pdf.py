import fitz
import re
import os
import json
from typing import List, Dict


def clean_text(text: str) -> str:
    
    # исправляем переносы слов типа: "обра-\nзование"
    text = re.sub(r"-\s*\n\s*", "", text)
    
    # заменяем переносы строк на пробел
    text = re.sub(r"\n+", " ", text)
    
    # убираем множественные пробелы
    text = re.sub(r"\s+", " ", text)
    
    # убираем пробелы в начале и конце
    text = text.strip()
    
    return text


def extract_text_with_metadata(pdf_path: str) -> List[Dict]:

    doc = fitz.open(pdf_path)
    
    pages = []
    
    for page_num, page in enumerate(doc):
        
        raw_text = page.get_text("text")
        
        cleaned_text = clean_text(raw_text)
        
        if len(cleaned_text) < 10:
            continue
        
        pages.append({
            "text": cleaned_text,
            "page": page_num + 1,
            "source": os.path.basename(pdf_path)
        })
    
    doc.close()
    
    return pages


def extract_all_pdfs_from_folder(folder_path: str) -> List[Dict]:
    
    all_pages = []
    
    if not os.path.exists(folder_path):
        raise Exception(f"Folder not found: {folder_path}")
    
    files = os.listdir(folder_path)
    
    pdf_files = [f for f in files if f.lower().endswith(".pdf")]
    
    print(f"Найдено PDF файлов: {len(pdf_files)}")
    
    for filename in pdf_files:
        
        pdf_path = os.path.join(folder_path, filename)
        
        print(f"Обработка: {filename}")
        
        pages = extract_text_with_metadata(pdf_path)
        
        all_pages.extend(pages)
        
        print(f"  Страниц извлечено: {len(pages)}")
    
    print(f"\nВсего страниц извлечено: {len(all_pages)}")
    
    return all_pages


if __name__ == "__main__":

    folder = "documents"

    pages = extract_all_pdfs_from_folder(folder)

    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print("Pages saved to data/pages.json")