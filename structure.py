"""
structure.py — Step 3 of the pipeline.

Takes the verbatim TOC transcription and asks Cerebras to organize it into a
hierarchy of chapters -> lessons with page ranges.

HARD CONSTRAINT (enforced by prompt + verified later in verify.py):
Cerebras is FORBIDDEN to edit, reword, translate, summarize, expand, drop or
invent content. The ONLY permitted change to any title is fixing an obvious
spelling / OCR typo. It is a structuring step, not a rewriting step.
"""

import os
import json
import re

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY"),
)

MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")


_RULES_EN = """You are a TABLE-OF-CONTENTS STRUCTURE EXTRACTOR.

You receive the raw OCR text of a book's table of contents. Your ONLY job is to
organize it into chapters, and the lessons inside each chapter, with page numbers.

ABSOLUTE RULES — read carefully:
1. Use every title EXACTLY as it appears in the source text (verbatim, same words, same order).
2. The ONLY change you may EVER make to a title is correcting an obvious spelling or OCR typo
   (e.g. a swapped letter). You may NOT reword, translate, summarize, expand, shorten, or invent.
3. Do NOT add any chapter or lesson that is not in the source. Do NOT omit any that is present.
4. Page numbers MUST be taken from the source. Never invent a page number. If a page number
   for an item is genuinely absent from the source, use null.
5. A "lesson" is an entry nested under a chapter. If the book has no sub-entries under a
   heading, that chapter simply has an empty "lessons" list.
6. Output ONLY a single valid JSON object. No markdown, no commentary."""

_RULES_AR = """أنت "مُنظِّم فهرس المحتويات".

تستلم النص المُستخرَج (OCR) لفهرس محتويات كتاب. مهمتك الوحيدة هي تنظيمه إلى فصول،
والدروس داخل كل فصل، مع أرقام الصفحات.

قواعد صارمة — اقرأها بعناية:
1. استخدم كل عنوان كما هو تماماً في النص المصدر (حرفياً، نفس الكلمات، نفس الترتيب).
2. التعديل الوحيد المسموح به على أي عنوان هو تصحيح خطأ إملائي أو خطأ OCR واضح فقط.
   يُمنع منعاً باتاً إعادة الصياغة أو الترجمة أو التلخيص أو الإضافة أو الاختصار أو الاختراع.
3. لا تُضِف أي فصل أو درس غير موجود في المصدر. ولا تحذف أي عنصر موجود.
4. أرقام الصفحات يجب أن تؤخذ من المصدر فقط. لا تخترع رقم صفحة أبداً. إذا كان رقم الصفحة
   غير موجود فعلاً في المصدر، استخدم null.
5. "الدرس" هو عنصر يندرج تحت فصل. إذا لم يكن لعنوان فصلٍ عناصرُ فرعية، فاجعل قائمة "lessons" فارغة.
6. أخرج كائن JSON واحداً صحيحاً فقط. بدون أي تنسيق markdown أو تعليقات."""

_SCHEMA = """Return JSON with EXACTLY this shape:

{
  "chapters": [
    {
      "title": "<chapter title, verbatim>",
      "start_page": <int or null>,
      "end_page": <int or null>,
      "lessons": [
        { "title": "<lesson title, verbatim>", "start_page": <int or null>, "end_page": <int or null> }
      ]
    }
  ]
}"""


def _call(system_msg: str, user_msg: str) -> str:
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 8000,
    }
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception as e:
        if "response_format" in str(e) or "json" in str(e).lower():
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("Cerebras did not return valid JSON.")


def _to_int(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        if m:
            return int(m.group(0))
    return None


def _derive_end_pages(chapters: list[dict]) -> None:
    """Fill missing end_page values deterministically from the next item's start.

    A lesson ends one page before the next lesson (within the book's flat reading
    order); the last lesson inherits its chapter's end. A chapter ends one page
    before the next chapter starts.
    """
    # Flatten lessons in document order to derive lesson end pages.
    flat = []
    for ch in chapters:
        for ls in ch.get("lessons", []):
            flat.append(ls)

    for i, ls in enumerate(flat):
        if ls.get("end_page") is None and ls.get("start_page") is not None:
            nxt = flat[i + 1]["start_page"] if i + 1 < len(flat) else None
            if nxt is not None and nxt > ls["start_page"]:
                ls["end_page"] = nxt - 1

    # Chapter end pages from next chapter's start, or from last lesson.
    for i, ch in enumerate(chapters):
        if ch.get("end_page") is None:
            nxt = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None
            if nxt is not None and ch.get("start_page") and nxt > ch["start_page"]:
                ch["end_page"] = nxt - 1
            else:
                # Fall back to the chapter's LAST lesson end (may be null for the
                # final TOC entry — the book continues past the table of contents).
                lessons = ch.get("lessons") or []
                if lessons:
                    ch["end_page"] = lessons[-1].get("end_page")
        # Derive chapter start from first lesson if absent.
        if ch.get("start_page") is None:
            starts = [l.get("start_page") for l in ch.get("lessons", []) if l.get("start_page") is not None]
            if starts:
                ch["start_page"] = min(starts)


def _normalize(chapters: list) -> list[dict]:
    clean = []
    for ch in chapters or []:
        if not isinstance(ch, dict):
            continue
        lessons = []
        for ls in ch.get("lessons") or []:
            if not isinstance(ls, dict):
                continue
            lessons.append({
                "title": (ls.get("title") or "").strip(),
                "start_page": _to_int(ls.get("start_page")),
                "end_page": _to_int(ls.get("end_page")),
            })
        clean.append({
            "title": (ch.get("title") or "").strip(),
            "start_page": _to_int(ch.get("start_page")),
            "end_page": _to_int(ch.get("end_page")),
            "lessons": lessons,
        })
    return clean


def structure_toc(transcription: str, lang: str) -> list[dict]:
    """Return a list of chapter dicts (each with nested lessons + page ranges)."""
    rules = _RULES_AR if lang == "ar" else _RULES_EN
    system = rules
    user = f"{_SCHEMA}\n\n--- SOURCE TABLE OF CONTENTS TEXT ---\n\n{transcription}"

    raw = _call(system, user)
    data = _parse_json(raw)
    chapters = _normalize(data.get("chapters", []))
    _derive_end_pages(chapters)
    return chapters
