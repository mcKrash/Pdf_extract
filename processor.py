import os
import time
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
        if '؀' <= c <= 'ۿ'
        or 'ﭐ' <= c <= '﷿'
        or 'ﹰ' <= c <= '﻿'
    )


def count_latin_letters(text: str) -> int:
    return sum(1 for c in text if c.isascii() and c.isalpha())


def response_language_matches(response: str, expected: str) -> bool:
    ar = count_arabic(response)
    en = count_latin_letters(response)
    if expected == "ar":
        # Must have substantial Arabic AND Arabic must be ≥ 3x the Latin letters
        # (Latin allowed only for chemical symbols like H2O, CO2)
        return ar > 50 and ar > en * 3
    return en > ar * 1.5


def build_prompt(chunk: str, chunk_num: int, total: int, lang: str, strict: bool = False):
    if lang == "ar":
        extra = ""
        if strict:
            extra = (
                "\n\n⛔ المحاولة السابقة فشلت لأنك كتبت بالإنجليزية. "
                "هذه آخر فرصة. اكتب 100% بالعربية. ممنوع كتابة أي كلمة إنجليزية.\n"
            )
        system_msg = (
            "أنت معلم خبير في المواد العلمية. مهمتك: تحليل النص العربي وإنتاج مادة دراسية شاملة بالعربية الفصحى. "
            "اكتب بأسلوب واضح ومنسّق وجذاب. استخدم العناوين والقوائم لتسهيل القراءة. "
            "ابذل قصارى جهدك لاستخراج أي محتوى تعليمي ممكن حتى لو كان النص فيه أخطاء أو رموز غريبة. "
            "حاول تخمين السياق وأكمل المعلومات من معرفتك العلمية إن كانت الكلمات الرئيسية واضحة. "
            "الاستثناء الوحيد للإنجليزية: الرموز الكيميائية والمعادلات الرياضية. "
            "لا تكتفِ بقول 'النص غير واضح' إلا إذا كان فعلاً لا يحتوي على أي كلمات مفهومة." + extra
        )
        user_msg = f"""🔴 تعليمات حاسمة: إجابتك يجب أن تكون 100% باللغة العربية. أي كلمة إنجليزية = فشل. الاستثناء الوحيد: الرموز الكيميائية مثل H₂O و CO₂.

النص من الكتاب (القسم {chunk_num} من {total}):

{chunk}

---

اكتب المادة الدراسية بالعربية الفصحى فقط. ابدأ مباشرة بـ "## 📚 القسم":

## 📚 القسم {chunk_num}: [العنوان بالعربية من النص، أو "نص غير واضح"]

### 🎯 أهداف التعلم
- [الهدف الأول بالعربية]
- [الهدف الثاني بالعربية]
- (حتى 5 أهداف)

### 📖 شرح الدرس
[شرح كامل بالعربية لما جاء في النص فقط]

### 🔑 المصطلحات الرئيسية
- [المصطلح بالعربية]: [التعريف بالعربية]

### ✅ التحقق العلمي
[التحقق بالعربية من المعادلات والصيغ]

### ❓ أسئلة الامتحان

**اختيار من متعدد (5 أسئلة):**
س1. [السؤال بالعربية]
أ) ... ب) ... ج) ... د) ...
الإجابة: [الحرف] — [السبب بالعربية]
(كرر حتى س5)

**إجابة قصيرة (3 أسئلة):**
س6. [السؤال بالعربية]
الإجابة: [الجواب بالعربية]
(حتى س8)

**حل مسائل (سؤالان إن وُجدت):**
س9. [المسألة بالعربية]
الحل: [الخطوات بالعربية]

🔴 تذكير أخير: كل كلمة بالعربية. ممنوع الإنجليزية إطلاقاً عدا الرموز الكيميائية. إذا النص غير واضح اكتب "النص غير واضح" ولا تخترع شيئاً.
"""
    else:
        extra = ""
        if strict:
            extra = "\n\nWARNING: Previous attempt was in Arabic. Write in English only."
        system_msg = (
            "You are an expert educator. Respond ONLY in English. "
            "Use ONLY the provided text — do not invent." + extra
        )
        user_msg = f"""RAW BOOK TEXT (Section {chunk_num} of {total}):

{chunk}

---

Produce in English only:

## 📚 Section {chunk_num}: [Title from text]

### 🎯 Learning Objectives
[3–5 points]

### 📖 Lesson Explanation
[Based only on the text]

### 🔑 Key Terms
[Term]: [Definition]

### ✅ Scientific Validation
[Verify formulas]

### ❓ Exam Q&A

**Multiple Choice (5):**
Q1. ... A) B) C) D) Answer: [X]

**Short Answer (3):**
Q6. ... Answer: ...

**Problem Solving (2 — if applicable):**
Q9. ... Solution: ...
"""

    return system_msg, user_msg


def _call(system_msg: str, user_msg: str) -> str:
    time.sleep(REQUEST_DELAY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise Exception("Empty response from model")
    return text


def translate_to_arabic(english_text: str) -> str:
    """Force-translate any English content to clean polished Arabic."""
    system = (
        "أنت مترجم أكاديمي محترف ومحرر لغوي. "
        "مهمتك: ترجمة وتنقيح النص التالي إلى العربية الفصحى المعاصرة بأسلوب جذاب ومنسّق. "
        "قواعد صارمة:\n"
        "1. كل كلمة بالعربية الفصحى، لا تترك أي كلمة إنجليزية.\n"
        "2. احتفظ بالتنسيق Markdown (العناوين ## و ### والقوائم والجداول).\n"
        "3. اجعل الأسلوب أكاديمياً واضحاً، تجنب الترجمة الحرفية الجافة.\n"
        "4. الاستثناءات المسموحة فقط: الرموز الكيميائية مثل H₂O و CO₂ والمعادلات الرياضية.\n"
        "5. أعد صياغة الجمل لتكون أكثر سلاسة وفصاحة عند الترجمة.\n"
        "6. لا تضف أي ملاحظات أو تعليقات خارج المحتوى.\n"
        "7. حافظ على نفس البنية: العنوان، الأهداف، الشرح، المصطلحات، التحقق، الأسئلة."
    )
    user = f"ترجم النص التالي إلى عربية فصحى منسّقة وجذابة:\n\n{english_text}"
    return _call(system, user)


def process_chunk(chunk: str, chunk_num: int, total: int, lang: str = "en") -> str:
    system_msg, user_msg = build_prompt(chunk, chunk_num, total, lang, strict=False)

    try:
        text = _call(system_msg, user_msg)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            time.sleep(10)
            text = _call(system_msg, user_msg)
        else:
            raise Exception(f"Cerebras error on section {chunk_num}: {err}")

    # If language matches, we're done
    if response_language_matches(text, lang):
        return text

    # SAFETY NET for Arabic books: force-translate the wrong-language response
    if lang == "ar":
        try:
            translated = translate_to_arabic(text)
            if count_arabic(translated) > 50:
                return translated
        except Exception:
            pass
        # Last resort — guaranteed Arabic-only fallback
        return (
            f"## 📚 القسم {chunk_num}: نص غير قابل للتحليل\n\n"
            "### ⚠️ ملاحظة\n"
            "تعذّر استخراج محتوى تعليمي واضح من هذا القسم. "
            "قد يكون النص الأصلي غير مقروء أو مشوشاً في ملف الـ PDF. "
            "يُرجى مراجعة الصفحات الأصلية في الكتاب.\n"
        )

    return text
