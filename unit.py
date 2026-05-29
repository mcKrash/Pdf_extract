"""
unit.py — full unit extraction: chapter -> lessons -> verbatim content -> questions.

Pipeline for ONE unit (given its PDF page range):
  1. OCR every page in the range            (ocr.transcribe_pages)
  2. Detect printed<->PDF page offset        (from page footers)
  3. Parse the unit's lesson listing         (Cerebras, structure only)
  4. Slice each lesson's VERBATIM content     (raw OCR text — LLM never touches it)
  5. Generate exam questions + model answers  (Cerebras, grounded in the content)

GUARANTEE: lesson `content` is the raw OCR transcription, untouched by any LLM.
Cerebras is used ONLY to (a) read the listing into structure and (b) write
questions grounded in the content. The book's text itself is never rewritten.
"""

import re
import json

from ocr import transcribe_pages
from structure import _call, _parse_json

# Entries in the listing that are not graded lessons (no questions generated).
_NON_LESSON = ["أهداف", "المقدمة", "مقدمة", "تمرين", "تمارين",
               "objectives", "introduction", "exercise", "review"]

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _to_western_digits(s: str) -> str:
    return s.translate(_AR_DIGITS)


# --------------------------------------------------------------------------
# 2. Offset detection
# --------------------------------------------------------------------------
def _footer_page_number(text: str) -> int | None:
    """Pull the printed page number from a page's footer, if present."""
    lines = [l.strip() for l in _to_western_digits(text).splitlines() if l.strip()]
    # Look at the last few lines (footers live at the bottom).
    for line in reversed(lines[-6:]):
        # Standalone number, optionally wrapped in dashes/spaces: "- 6 -", "13", "-5-"
        m = re.fullmatch(r"[-\s]*([0-9]{1,4})[-\s]*", line)
        if m:
            return int(m.group(1))
    return None


def detect_offset(pages: list[dict]) -> tuple[int, str | None]:
    """Return (offset, note). printed_page + offset = pdf_page."""
    candidates: list[int] = []
    for p in pages:
        printed = _footer_page_number(p["text"])
        if printed is not None:
            candidates.append(p["pdf_page"] - printed)
    if not candidates:
        return 0, "Could not detect page offset from footers; assuming printed == PDF page."
    # Most common offset wins (robust against stray footer misreads).
    offset = max(set(candidates), key=candidates.count)
    return offset, None


# --------------------------------------------------------------------------
# 3. Parse the unit lesson listing
# --------------------------------------------------------------------------
_LISTING_KEYS = ["رقم الدرس", "محتويات الوحدة", "الموضوع", "lesson", "contents"]

_LISTING_RULES_AR = """أنت تقرأ صفحة "محتويات الوحدة" من كتاب مدرسي. مهمتك تنظيمها فقط.

قواعد صارمة:
- استخدم العناوين حرفياً كما في النص (تصحيح إملائي فقط، ممنوع إعادة الصياغة/الترجمة/الإضافة/الحذف).
- أرقام الصفحات من المصدر فقط؛ إن غابت استخدم null.
- "lesson_number" = رقم الدرس النصي (الأول، الثاني...) أو null إن لم يكن درساً مرقماً (مثل: أهداف الوحدة، المقدمة، تمرين عام).
- "unit_title" = العنوان الوصفي للوحدة (مثل: "الكيمياء العضوية المشبعة الهيدروكربونية وتطبيقاتها")، وليس كلمة "محتويات الوحدة".
- الصيغ العلمية: أي صيغة كيميائية أو رياضية في أي حقل تُكتب كـ LaTeX inline: $H_2O$ و $C_nH_{2n+1}$ و $CH_3-CH_2-OH$.
  ممنوع Unicode للمرتفع/المنخفض. لا تُغيِّر المحتوى العلمي — فقط نسّقه.
- أعد JSON صحيحاً فقط بهذا الشكل:

{
  "unit_number": "الوحدة الأولى",
  "unit_title": "عنوان الوحدة الوصفي حرفياً",
  "entries": [
    {"lesson_number": "الأول|null", "title": "العنوان حرفياً", "printed_start": رقم|null}
  ]
}"""

_LISTING_RULES_EN = """You are reading a textbook unit's contents/listing page. Organize it ONLY.

Strict rules:
- Use titles verbatim (spelling fixes only; never reword/translate/add/drop).
- Page numbers from source only; null if absent.
- "lesson_number" = the lesson ordinal text or null for non-lessons (objectives, introduction, review/exercise).
- Scientific notation: any chemical formula or math expression in any field MUST be LaTeX inline:
  $H_2O$, $C_nH_{2n+1}$, $CH_3-CH_2-OH$. Never use Unicode subscripts/superscripts. Do not alter the content.
- Return ONLY valid JSON:

{
  "unit_number": "Unit One",
  "unit_title": "verbatim unit title",
  "entries": [
    {"lesson_number": "1|null", "title": "verbatim title", "printed_start": int|null}
  ]
}"""


def _find_listing_pages(pages: list[dict]) -> str:
    """Concatenate the unit divider/title pages + the listing page(s).

    The descriptive unit title (e.g. 'الكيمياء العضوية...') lives on the unit
    divider page, while page numbers live on the 'محتويات الوحدة' listing page —
    we feed both so the parser gets the real title AND the lesson page numbers.
    """
    head = [p["text"] for p in pages[:3]]
    listing = [p["text"] for p in pages if any(k in p["text"] for k in _LISTING_KEYS)]
    seen, out = set(), []
    for t in head + listing:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return "\n\n".join(out[:4])


def parse_listing(pages: list[dict], lang: str) -> dict:
    rules = _LISTING_RULES_AR if lang == "ar" else _LISTING_RULES_EN
    listing_text = _find_listing_pages(pages)
    raw = _call(rules, "نص صفحة المحتويات:\n\n" + listing_text if lang == "ar"
                else "Listing page text:\n\n" + listing_text)
    data = _parse_json(raw)
    entries = []
    for e in data.get("entries", []) or []:
        if not isinstance(e, dict):
            continue
        ps = e.get("printed_start")
        if isinstance(ps, str):
            m = re.search(r"\d+", _to_western_digits(ps))
            ps = int(m.group()) if m else None
        entries.append({
            "lesson_number": (e.get("lesson_number") or None),
            "title": (e.get("title") or "").strip(),
            "printed_start": ps if isinstance(ps, int) else None,
        })
    return {
        "unit_number": (data.get("unit_number") or "").strip(),
        "unit_title": (data.get("unit_title") or "").strip(),
        "entries": entries,
    }


# --------------------------------------------------------------------------
# 4. Slice verbatim content per lesson
# --------------------------------------------------------------------------
def _derive_printed_ends(entries: list[dict], unit_printed_end: int | None) -> None:
    starts = [e["printed_start"] for e in entries]
    for i, e in enumerate(entries):
        if e["printed_start"] is None:
            e["printed_end"] = None
            continue
        nxt = None
        for j in range(i + 1, len(entries)):
            if starts[j] is not None and starts[j] > e["printed_start"]:
                nxt = starts[j]
                break
        if nxt is not None:
            e["printed_end"] = nxt - 1
        else:
            e["printed_end"] = unit_printed_end
        # Guard against bad derivation (e.g. last entry beyond the processed range).
        if e["printed_end"] is not None and e["printed_end"] < e["printed_start"]:
            e["printed_end"] = e["printed_start"]


def _page_text_map(pages: list[dict]) -> dict[int, str]:
    return {p["pdf_page"]: p["text"] for p in pages}


def slice_content(pages: list[dict], offset: int, printed_start: int | None,
                  printed_end: int | None) -> str:
    """Concatenate the raw OCR text of the lesson's PDF pages (verbatim)."""
    if printed_start is None:
        return ""
    pmap = _page_text_map(pages)
    pdf_start = printed_start + offset
    pdf_end = (printed_end + offset) if printed_end is not None else pdf_start
    chunks = []
    for pdf_page in range(pdf_start, pdf_end + 1):
        if pdf_page in pmap and pmap[pdf_page]:
            chunks.append(pmap[pdf_page])
    return "\n\n".join(chunks).strip()


# --------------------------------------------------------------------------
# 5. Generate questions (grounded, with answers)
# --------------------------------------------------------------------------
_Q_RULES_AR = """أنت معلم كيمياء سوداني تضع أسئلة امتحان للصف الثالث الثانوي.

اعتمد حصرياً على نص الدرس المعطى أدناه. ممنوع منعاً باتاً اختراع معلومات غير موجودة
في النص، وممنوع تغيير أو إعادة صياغة نص الكتاب. الأسئلة جديدة لكن يجب أن تكون
مستندة بالكامل إلى محتوى الدرس. هذا منهج دراسي رسمي والطلاب يُمتحنون عليه، لذا
الدقة العلمية الحرفية إلزامية.

أنشئ أسئلة من هذه الأنواع (حسب ما يناسب الدرس):
- mcq: اختيار من متعدد بأربعة خيارات + الإجابة الصحيحة + شرح قصير.
- short: سؤال قصير + الإجابة النموذجية.
- essay: سؤال مقالي (اشرح/قارن/علّل) + الإجابة النموذجية.
- problem: مسألة/معادلة كيميائية (إن وُجدت تفاعلات أو حسابات) + الحل خطوة بخطوة.

قاعدة الصيغ العلمية — إلزامية في كل حقل (question, options, answer, explanation):
- كل صيغة كيميائية أو رياضية أو معادلة تُكتب حصراً كـ LaTeX inline داخل $...$
- أمثلة: H2O ← $H_2O$ | CnH2n+1 ← $C_nH_{2n+1}$ | CH3-CH2-OH ← $CH_3-CH_2-OH$
- معادلة: 2H2 + O2 → 2H2O تصبح $2H_2 + O_2 \\rightarrow 2H_2O$
- ممنوع منعاً باتاً Unicode للمرتفع/المنخفض (H₂O ✗ — $H_2O$ ✓)
- لا تُغيِّر القيم العلمية أبداً — فقط نسّقها بـ LaTeX

أعد JSON صحيحاً فقط بهذا الشكل:
{
  "questions": [
    {"type":"mcq","question":"...","options":["أ) ...","ب) ...","ج) ...","د) ..."],"answer":"...","explanation":"..."},
    {"type":"short","question":"...","answer":"..."},
    {"type":"essay","question":"...","answer":"..."},
    {"type":"problem","question":"...","answer":"..."}
  ]
}
كل النصوص بالعربية الفصحى."""

_Q_RULES_EN = """You are a chemistry teacher writing secondary-school exam questions.

Use ONLY the lesson text below. NEVER invent facts not in the text, and never
reword the book. Questions are new but must be fully grounded in the lesson.
This is official curriculum — students are examined on it, so scientific accuracy
must be exact and literal.

Generate these types as suitable: mcq (4 options + answer + short explanation),
short (Q + model answer), essay (explain/compare/justify + model answer),
problem (chemical equations/calculations + step-by-step solution).

SCIENTIFIC NOTATION RULE — mandatory in every field (question, options, answer, explanation):
- Every chemical formula, equation, or math expression MUST be LaTeX inline: $...$
- Examples: H2O → $H_2O$ | CnH2n+1 → $C_nH_{2n+1}$ | CH3-CH2-OH → $CH_3-CH_2-OH$
- Equations: 2H2 + O2 → 2H2O becomes $2H_2 + O_2 \\rightarrow 2H_2O$
- NEVER use Unicode subscripts/superscripts (H₂O ✗ — $H_2O$ ✓)
- Do NOT change the scientific values themselves — only format them as LaTeX

Return ONLY valid JSON:
{
  "questions": [
    {"type":"mcq","question":"...","options":["A) ...","B) ...","C) ...","D) ..."],"answer":"...","explanation":"..."},
    {"type":"short","question":"...","answer":"..."},
    {"type":"essay","question":"...","answer":"..."},
    {"type":"problem","question":"...","answer":"..."}
  ]
}"""

# Cap content sent to the model (grounding stays faithful; full text is stored separately).
_MAX_CONTENT_CHARS = 12000


def generate_questions(content: str, lang: str, types: list[str], count: int) -> list[dict]:
    if not content.strip():
        return []
    rules = _Q_RULES_AR if lang == "ar" else _Q_RULES_EN
    grounded = content[:_MAX_CONTENT_CHARS]
    types_str = ", ".join(types)
    if lang == "ar":
        user = (f"الأنواع المطلوبة: {types_str}. أنشئ حوالي {count} سؤالاً موزعة على هذه الأنواع.\n\n"
                f"نص الدرس:\n\n{grounded}")
    else:
        user = (f"Required types: {types_str}. Generate about {count} questions across them.\n\n"
                f"LESSON TEXT:\n\n{grounded}")
    try:
        raw = _call(rules, user)
        data = _parse_json(raw)
    except Exception:
        return []
    out = []
    for q in data.get("questions", []) or []:
        if not isinstance(q, dict) or not (q.get("question") or "").strip():
            continue
        qt = (q.get("type") or "short").lower()
        if types and qt not in types:
            continue
        item = {"type": qt, "question": q["question"].strip(),
                "answer": (q.get("answer") or "").strip()}
        if q.get("options"):
            item["options"] = [str(o).strip() for o in q["options"]]
        if q.get("explanation"):
            item["explanation"] = str(q["explanation"]).strip()
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def _is_lesson(entry: dict) -> bool:
    if entry.get("lesson_number"):
        return True
    title = entry.get("title", "")
    return not any(k in title for k in _NON_LESSON)


def extract_unit(pdf_path: str, pdf_start_page: int, pdf_end_page: int, lang: str,
                 q_types: list[str], q_count: int) -> dict:
    pages = transcribe_pages(pdf_path, pdf_start_page, pdf_end_page, lang)
    if not any(p["text"] for p in pages):
        raise ValueError("OCR produced no text for the given PDF page range.")

    offset, offset_note = detect_offset(pages)
    listing = parse_listing(pages, lang)
    entries = listing["entries"]

    # Unit's printed end = last page in range minus offset (for end-page derivation).
    unit_printed_end = pdf_end_page - offset
    _derive_printed_ends(entries, unit_printed_end)

    warnings: list[str] = []
    if offset_note:
        warnings.append(offset_note)
    if not entries:
        warnings.append("No lesson entries parsed from the unit listing.")

    lessons = []
    for e in entries:
        content = slice_content(pages, offset, e["printed_start"], e.get("printed_end"))
        is_lesson = _is_lesson(e)
        questions = []
        if is_lesson and content:
            questions = generate_questions(content, lang, q_types, q_count)
        elif is_lesson and not content:
            warnings.append(f"No content found for lesson: \"{e['title']}\" "
                            f"(printed {e['printed_start']}).")
        lessons.append({
            "lesson_number": e.get("lesson_number"),
            "title": e["title"],
            "is_lesson": is_lesson,
            "printed_start": e["printed_start"],
            "printed_end": e.get("printed_end"),
            "pdf_start": (e["printed_start"] + offset) if e["printed_start"] else None,
            "pdf_end": (e["printed_end"] + offset) if e.get("printed_end") else None,
            "content": content,            # VERBATIM raw OCR — never altered by an LLM
            "content_verified": bool(content),
            "questions": questions,
        })

    return {
        "language": lang,
        "page_offset": offset,
        "unit_number": listing["unit_number"],
        "unit_title": listing["unit_title"],
        "pdf_start_page": pdf_start_page,
        "pdf_end_page": pdf_end_page,
        "lessons": lessons,
        "warnings": warnings,
    }
