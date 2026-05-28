import fitz  # pymupdf
import unicodedata

MAX_WORDS = 400

JUNK_PATTERNS = [
    'isbn', 'www.', 'http', '©', 'copyright', 'all rights reserved',
    'حقوق الطبع', 'جميع الحقوق', 'الطبعة', 'المملكة العربية', 'دار النشر',
    'رقم الإيداع', 'فهرسة', 'بيانات الفهرسة', 'rowad', 'tamayoz', 'wizara',
    'المحتويات', 'الفهرس', 'فهرس المحتويات', 'table of contents', 'contents'
]


def is_toc_page(text: str) -> bool:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 5:
        return False
    # TOC lines typically end with a number (page number)
    number_endings = sum(
        1 for l in lines
        if l and l.split()[-1].isdigit()
    )
    return number_endings / len(lines) > 0.45


def is_content_page(text: str) -> bool:
    words = text.split()
    if len(words) < 15:
        return False
    if is_toc_page(text):
        return False
    text_lower = text.lower()
    # Any single TOC/copyright marker keyword kills the page
    strong_junk = ['المحتويات', 'الفهرس', 'table of contents', 'isbn', 'حقوق الطبع']
    if any(p in text_lower for p in strong_junk):
        return False
    junk_hits = sum(1 for p in JUNK_PATTERNS if p in text_lower)
    if junk_hits >= 2:
        return False
    return True


def detect_book_language(full_text: str) -> str:
    arabic = sum(
        1 for c in full_text
        if '؀' <= c <= 'ۿ' or 'ﭐ' <= c <= '﷿' or 'ﹰ' <= c <= '﻿'
    )
    latin = sum(1 for c in full_text if c.isascii() and c.isalpha())
    return "ar" if arabic > latin * 0.3 else "en"


def extract_pdf(pdf_path: str, job_id: str, max_pages: int = 0) -> tuple[list[str], str]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    limit = max_pages if max_pages > 0 else total_pages

    all_raw_text = []
    pages_text = []
    for i, page in enumerate(doc):
        if i >= limit:
            break
        text = unicodedata.normalize("NFKC", page.get_text("text")).strip()
        if text:
            all_raw_text.append(text)
            if is_content_page(text):
                pages_text.append(text)

    doc.close()

    if not pages_text:
        raise Exception("No content pages found. The PDF may be fully scanned or have no extractable text.")

    # Detect language once across the entire book (covers chemical-symbol chunks)
    book_lang = detect_book_language("\n".join(all_raw_text))

    full_text = "\n\n".join(pages_text)
    return split_into_chunks(full_text), book_lang


def is_quality_chunk(text: str) -> bool:
    """Reject only chunks that are pure symbols/numbers with no real words."""
    if not text or len(text) < 100:
        return False

    arabic = sum(
        1 for c in text
        if '؀' <= c <= 'ۿ' or 'ﭐ' <= c <= '﷿' or 'ﹰ' <= c <= '﻿'
    )
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    letters = arabic + latin

    # Need at least 80 readable letters (AR or EN) to be worth analyzing
    if letters < 80:
        return False

    total_meaningful = sum(1 for c in text if not c.isspace())
    if total_meaningful == 0:
        return False

    # At least 15% of non-whitespace chars must be actual letters
    return letters / total_meaningful >= 0.15


def split_into_chunks(content: str) -> list[str]:
    lines = content.split("\n")
    raw_chunks = []
    current = []

    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and len(stripped) < 80
            and current
            and any(kw in stripped for kw in [
                "chapter", "section", "unit",
                "الفصل", "الوحدة", "الباب", "الدرس", "درس", "فصل", "وحدة"
            ])
        ):
            text = "\n".join(current).strip()
            if len(text) > 200:
                raw_chunks.append(text)
            current = [line]
        else:
            current.append(line)

    if current:
        text = "\n".join(current).strip()
        if len(text) > 200:
            raw_chunks.append(text)

    if not raw_chunks and content.strip():
        raw_chunks = [content.strip()]

    # Enforce MAX_WORDS per chunk
    final_chunks = []
    for chunk in raw_chunks:
        words = chunk.split()
        if len(words) <= MAX_WORDS:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(words), MAX_WORDS):
                piece = " ".join(words[i: i + MAX_WORDS]).strip()
                if piece:
                    final_chunks.append(piece)

    # Quality filter: drop garbage chunks (too many symbols, not enough letters)
    final_chunks = [c for c in final_chunks if is_quality_chunk(c)]

    return final_chunks
