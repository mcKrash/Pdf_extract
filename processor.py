import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.getenv("CEREBRAS_API_KEY"),
)

MODEL = "gpt-oss-120b"
REQUEST_DELAY = 2


def count_arabic(text: str) -> int:
    return sum(
        1 for c in text
        if '؀' <= c <= 'ۿ' or 'ﭐ' <= c <= '﷿' or 'ﹰ' <= c <= '﻿'
    )


def count_latin_letters(text: str) -> int:
    return sum(1 for c in text if c.isascii() and c.isalpha())


def _call(system_msg: str, user_msg: str, force_json: bool = True) -> str:
    time.sleep(REQUEST_DELAY)
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
    }
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise Exception("Empty response from model")
    return text


def detect_subject(text: str) -> str:
    """Quick keyword-based subject hint."""
    t = text.lower()
    chem_kw = ["chemistry", "كيمياء", "ch3", "ch₃", "h2o", "mole", "atom", "molecule", "جزيء", "ذرة", "تفاعل"]
    phys_kw = ["physics", "فيزياء", "force", "energy", "newton", "velocity", "قوة", "طاقة", "سرعة"]
    math_kw = ["math", "رياضيات", "equation", "integral", "derivative", "معادلة", "تكامل", "مشتقة", "هندسة"]
    bio_kw = ["biology", "أحياء", "بيولوجيا", "cell", "خلية", "dna", "evolution", "تطور"]
    counts = {
        "chemistry": sum(1 for k in chem_kw if k in t),
        "physics": sum(1 for k in phys_kw if k in t),
        "math": sum(1 for k in math_kw if k in t),
        "biology": sum(1 for k in bio_kw if k in t),
    }
    best = max(counts.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "general"


JSON_SCHEMA_AR = """قم بإرجاع كائن JSON بهذه البنية بالضبط (كل الحقول النصية بالعربية الفصحى):

{
  "title": "عنوان الدرس من النص",
  "subject": "chemistry | physics | math | biology | general",
  "estimated_reading_minutes": رقم تقديري (3-15),
  "learning_objectives": ["هدف 1", "هدف 2", "هدف 3"],
  "lesson_explanation": "شرح مفصّل وواضح بالعربية",
  "key_terms": [
    {"term": "المصطلح بالعربية", "definition": "التعريف بالعربية"}
  ],
  "scientific_validation": {
    "is_valid": true,
    "notes": "تعليق على صحة المعادلات والصيغ بالعربية"
  },
  "exam": {
    "multiple_choice": [
      {"question": "السؤال", "options": ["أ) ...", "ب) ...", "ج) ...", "د) ..."], "answer": "ب", "explanation": "السبب بالعربية"}
    ],
    "short_answer": [
      {"question": "السؤال", "answer": "الإجابة"}
    ],
    "problem_solving": [
      {"problem": "المسألة", "solution": "الحل خطوة بخطوة"}
    ]
  }
}

قواعد:
- اكتب كل النصوص بالعربية الفصحى (الاستثناء الوحيد: الرموز الكيميائية مثل H₂O والمعادلات الرياضية).
- 5 أسئلة اختيار من متعدد، 3 إجابة قصيرة، 0-2 حل مسائل (فقط إن وُجدت في النص).
- استخدم محتوى النص فقط، لا تخترع.
- إذا النص غير واضح، أعد JSON بـ title="نص غير قابل للتحليل" و learning_objectives=[] والباقي قصير.
"""

JSON_SCHEMA_EN = """Return a JSON object with EXACTLY this structure (all text fields in English):

{
  "title": "Lesson title from the text",
  "subject": "chemistry | physics | math | biology | general",
  "estimated_reading_minutes": estimated number (3-15),
  "learning_objectives": ["objective 1", "objective 2", "objective 3"],
  "lesson_explanation": "Detailed clear explanation",
  "key_terms": [
    {"term": "Term", "definition": "Definition"}
  ],
  "scientific_validation": {
    "is_valid": true,
    "notes": "Notes on formula/equation correctness"
  },
  "exam": {
    "multiple_choice": [
      {"question": "Question?", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "B", "explanation": "Why this is correct"}
    ],
    "short_answer": [
      {"question": "Question?", "answer": "Answer"}
    ],
    "problem_solving": [
      {"problem": "Problem", "solution": "Step-by-step solution"}
    ]
  }
}

Rules:
- All text in English (exception: chemical symbols, math formulas).
- 5 multiple choice, 3 short answer, 0-2 problem solving (only if present in text).
- Use ONLY content from the text. Do not invent.
- If text is unclear, return title="Text not analyzable" and minimal content.
"""


def build_messages(chunk: str, lang: str, strict: bool = False):
    if lang == "ar":
        system = (
            "أنت معلم خبير في المواد العلمية، تحلل النصوص العربية وتنتج مادة دراسية منظمة. "
            "أعد فقط JSON صالح بدون أي نص خارجه. "
            "كل القيم النصية يجب أن تكون بالعربية الفصحى (الاستثناء: الرموز الكيميائية والرياضية)."
            + ("\n⛔ المحاولة السابقة احتوت إنجليزية. اكتب كل القيم النصية بالعربية فقط." if strict else "")
        )
        user = f"النص من الكتاب:\n\n{chunk}\n\n---\n\n{JSON_SCHEMA_AR}"
    else:
        system = (
            "You are an expert science educator. Analyze the text and produce structured study material. "
            "Return ONLY valid JSON, nothing else. All text values must be in English."
            + ("\nWARNING: Previous attempt was in Arabic. Use English only." if strict else "")
        )
        user = f"BOOK TEXT:\n\n{chunk}\n\n---\n\n{JSON_SCHEMA_EN}"

    return system, user


def parse_json(raw: str) -> dict | None:
    """Best-effort JSON extraction from model output."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try to find a JSON block within the text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def language_ok(obj: dict, lang: str) -> bool:
    """Check that the JSON object's text content is in the expected language."""
    flat = json.dumps(obj, ensure_ascii=False)
    ar = count_arabic(flat)
    en = count_latin_letters(flat)
    if lang == "ar":
        return ar > 50 and ar > en * 2
    return en > ar


def translate_obj_to_arabic(obj: dict) -> dict | None:
    raw = json.dumps(obj, ensure_ascii=False)
    system = (
        "أنت مترجم أكاديمي. ترجم كل القيم النصية في JSON التالي إلى عربية فصحى منسّقة. "
        "حافظ على بنية JSON تماماً. لا تترجم المفاتيح. لا تغيّر القيم البولية أو الأرقام. "
        "اترك فقط الرموز الكيميائية والمعادلات الرياضية بالإنجليزية. "
        "أعد فقط JSON صالح."
    )
    user = f"ترجم القيم النصية فقط:\n\n{raw}"
    try:
        result = _call(system, user)
        return parse_json(result)
    except Exception:
        return None


def process_chunk(chunk: str, chunk_num: int, total: int, lang: str = "en") -> dict:
    """Returns a structured dict (JSON object) ready for DB storage."""
    system, user = build_messages(chunk, lang, strict=False)

    try:
        raw = _call(system, user)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            time.sleep(10)
            raw = _call(system, user)
        else:
            raise Exception(f"Cerebras error on section {chunk_num}: {err}")

    obj = parse_json(raw)

    if obj and language_ok(obj, lang):
        obj["section_number"] = chunk_num
        obj["language"] = lang
        if "subject" not in obj or obj.get("subject") == "general":
            obj["subject"] = detect_subject(chunk)
        return obj

    # Safety net: translate to Arabic if needed
    if lang == "ar" and obj:
        translated = translate_obj_to_arabic(obj)
        if translated and language_ok(translated, "ar"):
            translated["section_number"] = chunk_num
            translated["language"] = "ar"
            if "subject" not in translated or translated.get("subject") == "general":
                translated["subject"] = detect_subject(chunk)
            return translated

    # Last-resort fallback (guaranteed correct-language placeholder)
    if lang == "ar":
        return {
            "section_number": chunk_num,
            "language": "ar",
            "title": "نص غير قابل للتحليل",
            "subject": detect_subject(chunk),
            "estimated_reading_minutes": 0,
            "learning_objectives": [],
            "lesson_explanation": "تعذّر استخراج محتوى تعليمي واضح من هذا القسم. قد يكون النص الأصلي مشوّشاً.",
            "key_terms": [],
            "scientific_validation": {"is_valid": False, "notes": ""},
            "exam": {"multiple_choice": [], "short_answer": [], "problem_solving": []},
        }
    return {
        "section_number": chunk_num,
        "language": "en",
        "title": "Text not analyzable",
        "subject": detect_subject(chunk),
        "estimated_reading_minutes": 0,
        "learning_objectives": [],
        "lesson_explanation": "Could not extract a clear lesson from this section.",
        "key_terms": [],
        "scientific_validation": {"is_valid": False, "notes": ""},
        "exam": {"multiple_choice": [], "short_answer": [], "problem_solving": []},
    }
