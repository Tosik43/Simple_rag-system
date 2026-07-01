import re
import pymorphy3
import json
from typing import List, Dict


morph = pymorphy3.MorphAnalyzer()


def split_into_sentences(text: str) -> List[str]:

    sentences = re.split(r'(?<=[.!?])\s+', text)

    return [s.strip() for s in sentences if s.strip()]


def russian_ratio(text: str) -> float:

    russian_letters = sum(
        c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
             "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        for c in text
    )

    letters = sum(c.isalpha() for c in text)

    if letters == 0:
        return 0

    return russian_letters / letters


def russian_word_ratio(text: str) -> float:

    words = re.findall(r"[а-яА-ЯёЁ]{3,}", text)

    if not words:
        return 0

    valid_words = 0

    for word in words:

        parsed = morph.parse(word)

        if (
            parsed
            and parsed[0].is_known
            and parsed[0].score > 0.2
        ):
            valid_words += 1

    return valid_words / len(words)



def clean_chunk_text(text: str) -> str:

    text = re.sub(r'Страница\s+\d+\s+из\s+\d+', '', text, flags=re.IGNORECASE)

    text = re.sub(r'Документ зарегистрирован №.*?\d{4}', '', text)

    text = re.sub(r'\d{2}\.\d{2}\.\d{4}', '', text)

    text = re.sub(r'\s+\d+\s*$', '', text)

    text = re.sub(r'[‘’“”]', '', text)

    text = re.sub(r'[-–—]{2,}', '-', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()



def is_valid_chunk(text: str) -> tuple[bool, str]:
    """
    Проверяет пригодность чанка для индексации.
    Возвращает:
        (True, "")
        (False, причина)
    """

    text = text.strip()

    if len(text) < 80:
        return False, "слишком короткий"

    words = text.split()

    if len(words) < 10:
        return False, "мало слов"

    if russian_ratio(text) < 0.3:
        return False, "слишком мало русского текста"

    letters = sum(c.isalpha() for c in text)

    if letters == 0:
        return False, "нет букв"

    bad_chars = sum(
        not c.isalnum() and c not in " .,!?()-–:;%№«»\"'\n"
        for c in text
    )

    if bad_chars / len(text) > 0.3:
        return False, "слишком много спецсимволов"

    return True, ""



def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap_sentences: int = 2
) -> List[str]:

    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        if sentence_length > chunk_size:
            sentence = sentence[:chunk_size]
            sentence_length = len(sentence)

        if current_length + sentence_length > chunk_size:

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap = current_chunk[-overlap_sentences:]

            current_chunk = overlap.copy()
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def chunk_pages(pages: List[Dict]) -> List[Dict]:

    all_chunks = []
    skipped = 0

    for page in pages:

        text_chunks = chunk_text(page["text"])

        for i, chunk in enumerate(text_chunks):

            chunk = clean_chunk_text(chunk)

            valid, reason = is_valid_chunk(chunk)

            if not valid:
                skipped += 1

                print(
                    f"[SKIP] {page['source']} "
                    f"page={page['page']} "
                    f"chunk={i} "
                    f"reason={reason}"
                )

                continue

            all_chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": f"{page['source']}_page{page['page']}_chunk{i}"
            })

    print(f"\nСоздано chunks: {len(all_chunks)}")
    print(f"Пропущено chunks: {skipped}")

    return all_chunks

if __name__ == "__main__":

    with open("data/pages.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages)

    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)