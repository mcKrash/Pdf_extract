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
    base_kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
    }

    if force_json:
        try:
            kwargs = {**base_kwargs, "response_format": {"type": "json_object"}}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            if text.strip():
                return text
        except Exception as e:
            # API may not support response_format — retry without it
            if "response_format" not in str(e) and "json" not in str(e).lower():
                # Different error — re-raise
                raise

    response = client.chat.completions.create(**base_kwargs)
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
  "title": "عنوان الدرس",
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

قواعد مهمة جداً:
1. هذه المادة لامتحان طلاب، فلا يجوز ترك أي قسم بدون محتوى تعليمي.
2. حتى لو كان النص فيه أخطاء OCR أو رموز غريبة، استخرج الكلمات المفتاحية المفهومة وأكمل الدرس من معرفتك العلمية.
3. ابحث في النص عن أي مصطلح علمي أو كيميائي أو فيزيائي أو رياضي (مثل: ألكان، هاليد، حمض، تفاعل، ذرة، إلكترون، طاقة، معادلة، دالة) واكتب درساً كاملاً حوله.
4. اكتب 5 أسئلة اختيار من متعدد، 3 إجابة قصيرة، و1-2 حل مسائل.
5. كل النصوص بالعربية الفصحى (الاستثناء فقط: الرموز الكيميائية H₂O، CO₂ والمعادلات الرياضية).
6. ممنوع منعاً باتاً كتابة "نص غير قابل للتحليل" أو ما شابه — يجب إنتاج درس حقيقي دائماً.
"""

JSON_SCHEMA_EN = """Return a JSON object with EXACTLY this structure (all text fields in English):

{
  "title": "Lesson title",
  "subject": "chemistry | physics | math | biology | general",
  "estimated_reading_minutes": 3-15,
  "learning_objectives": ["objective 1", "objective 2", "objective 3"],
  "lesson_explanation": "Detailed clear explanation",
  "key_terms": [{"term": "Term", "definition": "Definition"}],
  "scientific_validation": {"is_valid": true, "notes": "Notes on correctness"},
  "exam": {
    "multiple_choice": [{"question": "?", "options": ["A) ...","B) ...","C) ...","D) ..."], "answer": "B", "explanation": "..."}],
    "short_answer": [{"question": "?", "answer": "..."}],
    "problem_solving": [{"problem": "...", "solution": "..."}]
  }
}

CRITICAL RULES:
1. This is exam-prep material — NEVER leave a section empty or unanalyzed.
2. Even if the text has OCR errors or strange symbols, extract any recognizable scientific keywords and use your knowledge to write a complete lesson around them.
3. Look for ANY scientific term (alkane, acid, reaction, atom, energy, equation, function...) and build a full lesson on it.
4. Always produce 5 multiple choice, 3 short answer, 1-2 problem solving questions.
5. English only (exception: chemical symbols and math formulas).
6. FORBIDDEN: writing "Text not analyzable" or similar — always produce a real lesson.
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


def extract_keywords(chunk: str) -> str:
    """Pull all recognizable scientific keywords from a possibly-garbled chunk."""
    words = re.findall(r'[؀-ۿ]+|[A-Za-z]{3,}|\d+[A-Za-z]+|[A-Za-z]+\d+', chunk)
    # Filter out very short / noisy tokens
    return " ".join(w for w in words if len(w) >= 3)[:1500]


def _generate_section(chunk: str, lang: str, chunk_num: int, force_message: str = "") -> dict | None:
    """Single generation attempt. Returns parsed obj or None."""
    system, user = build_messages(chunk, lang, strict=False)
    if force_message:
        user = force_message + "\n\n" + user
    try:
        raw = _call(system, user)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            time.sleep(10)
            try:
                raw = _call(system, user)
            except Exception:
                return None
        else:
            return None
    return parse_json(raw)


def process_chunk(chunk: str, chunk_num: int, total: int, lang: str = "en") -> dict:
    """Returns a structured dict (JSON object) ready for DB storage."""
    # Attempt 1: normal generation
    obj = _generate_section(chunk, lang, chunk_num)

    # Attempt 2: if model gave up, retry with just the keywords + a hard demand
    if not obj or _is_empty_obj(obj):
        keywords = extract_keywords(chunk)
        if lang == "ar":
            force = (
                "⚠️ المحاولة الأولى لم تنتج محتوى. أنت الآن مطالب بإنتاج درس كامل "
                "استناداً إلى الكلمات المفتاحية التالية المستخرجة من الكتاب. "
                "استخدم معرفتك العلمية لبناء درس متكامل حول هذه الكلمات. "
                f"الكلمات المفتاحية: {keywords}\n\n"
                "يجب أن يحتوي ردك على عنوان درس واقعي، 3-5 أهداف، شرح كامل، 5+ مصطلحات، "
                "5 أسئلة اختيار، 3 إجابة قصيرة، ومسألة واحدة على الأقل."
            )
        else:
            force = (
                "WARNING: First attempt produced no content. You MUST now produce a full lesson "
                "based on these keywords extracted from the book. Use your scientific knowledge "
                f"to build a complete lesson. Keywords: {keywords}\n\n"
                "Output must have a real lesson title, 3-5 objectives, full explanation, 5+ terms, "
                "5 multiple choice, 3 short answer, and at least 1 problem."
            )
        obj = _generate_section(chunk, lang, chunk_num, force)

    # Attempt 3: translation safety net for Arabic
    if obj and not language_ok(obj, lang) and lang == "ar":
        translated = translate_obj_to_arabic(obj)
        if translated:
            obj = translated

    if not obj:
        # Absolute last resort — minimal placeholder (still in correct language)
        return _emergency_placeholder(chunk, chunk_num, lang)

    obj["section_number"] = chunk_num
    obj["language"] = lang
    if "subject" not in obj or obj.get("subject") in (None, "general"):
        obj["subject"] = detect_subject(chunk)
    return obj


def _is_empty_obj(obj: dict) -> bool:
    """Detect 'gave up' responses where the model refused to produce content."""
    if not obj:
        return True
    title = (obj.get("title") or "").lower()
    bad_titles = ["text not analyzable", "نص غير قابل", "untitled", "no title", "غير واضح"]
    if any(b in title for b in bad_titles):
        return True
    if not obj.get("learning_objectives") and not obj.get("lesson_explanation"):
        return True
    explanation = obj.get("lesson_explanation", "")
    if len(explanation) < 100:
        return True
    return False


def _emergency_placeholder(chunk: str, chunk_num: int, lang: str) -> dict:
    subj = detect_subject(chunk)
    if lang == "ar":
        return {
            "section_number": chunk_num,
            "language": "ar",
            "title": f"مراجعة عامة - القسم {chunk_num}",
            "subject": subj,
            "estimated_reading_minutes": 5,
            "learning_objectives": ["مراجعة المفاهيم الأساسية في هذا القسم"],
            "lesson_explanation": "تعذّر تحليل تفاصيل هذا القسم تلقائياً. يُرجى مراجعة الصفحات الأصلية في الكتاب للحصول على المحتوى الكامل.",
            "key_terms": [],
            "scientific_validation": {"is_valid": True, "notes": ""},
            "exam": {"multiple_choice": [], "short_answer": [], "problem_solving": []},
        }
    return {
        "section_number": chunk_num,
        "language": "en",
        "title": f"General Review - Section {chunk_num}",
        "subject": subj,
        "estimated_reading_minutes": 5,
        "learning_objectives": ["Review the core concepts in this section"],
        "lesson_explanation": "Automated analysis did not produce detailed content. Refer to the original book pages.",
        "key_terms": [],
        "scientific_validation": {"is_valid": True, "notes": ""},
        "exam": {"multiple_choice": [], "short_answer": [], "problem_solving": []},
    }
