"""
ocr.py — Step 1 + 2 of the pipeline.

1. Render the chosen Table-of-Contents pages of a PDF to images (PyMuPDF).
2. Run an OCR backend over each image to get a VERBATIM text transcription.

The OCR backend is pluggable via the OCR_BACKEND env var:
    OCR_BACKEND=paddle   -> local PaddleOCR        (default, no API key)
    OCR_BACKEND=easyocr  -> local EasyOCR          (no API key)
    OCR_BACKEND=gemini   -> Google Gemini vision   (needs GEMINI_API_KEY)
    OCR_BACKEND=openai   -> OpenAI GPT-4o vision    (needs OPENAI_API_KEY)

Whatever the backend, the output is plain text that is later handed to
Cerebras for structuring. No backend is allowed to *interpret* the content —
it only transcribes what is on the page.
"""

import os
import io
import base64
import functools

# paddlepaddle 3.x crashes on Windows CPU through its oneDNN/PIR path
# ("ConvertPirAttribute2RuntimeAttribute not support"). Disabling mkldnn at the
# process level avoids it. MUST be set before any paddle import — and force it
# (not setdefault) so a stray pre-existing value can't leave mkldnn enabled.
os.environ["FLAGS_use_mkldnn"] = "0"

import fitz  # PyMuPDF
from PIL import Image

RENDER_DPI = int(os.getenv("RENDER_DPI", "200"))


# --------------------------------------------------------------------------
# Rendering: PDF pages -> PIL images
# --------------------------------------------------------------------------
def render_pages(pdf_path: str, start_page: int, end_page: int) -> list[Image.Image]:
    """Render 1-based inclusive page range [start_page, end_page] to images."""
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        if start_page < 1 or end_page < 1:
            raise ValueError("Page numbers are 1-based and must be >= 1.")
        if start_page > end_page:
            raise ValueError("toc_start_page must be <= toc_end_page.")
        if end_page > total:
            raise ValueError(
                f"toc_end_page ({end_page}) exceeds the PDF page count ({total})."
            )

        zoom = RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        images: list[Image.Image] = []
        for page_no in range(start_page, end_page + 1):
            page = doc[page_no - 1]  # 0-based internally
            pix = page.get_pixmap(matrix=matrix)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
        return images
    finally:
        doc.close()


def _img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------
# OCR backends
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=2)
def _paddle_reader(lang: str):
    """Cache one PaddleOCR instance per language (model load is expensive)."""
    from paddleocr import PaddleOCR

    # PaddleOCR language codes: 'arabic' covers Arabic + Latin + digits; 'en' for English.
    paddle_lang = "arabic" if lang == "ar" else "en"
    # Only `lang` is portable across 2.x/3.x. The oneDNN crash is avoided by the
    # FLAGS_use_mkldnn=0 env var set at the top of this module (before paddle imports),
    # NOT by a constructor arg (3.x rejects enable_mkldnn).
    try:
        return PaddleOCR(lang=paddle_lang)
    except Exception:
        return PaddleOCR()


def _ocr_paddle(img: Image.Image, lang: str) -> str:
    import numpy as np

    reader = _paddle_reader(lang)
    arr = np.array(img)  # RGB HWC

    # 3.x: .ocr() and .predict() both return the new "rec_texts" format.
    # 2.x: .ocr() returns the legacy nested [box, (text, conf)] format.
    # _parse_paddle handles both.
    try:
        result = reader.ocr(arr)
    except TypeError:
        result = reader.ocr(arr, cls=True)
    return _parse_paddle(result)


def _parse_paddle(result) -> str:
    """Parse PaddleOCR output, supporting both 3.x (rec_texts) and 2.x (nested)."""
    lines: list[str] = []

    for res in result or []:
        # --- 3.x format: dict / object exposing 'rec_texts' ---
        texts = None
        if isinstance(res, dict):
            texts = res.get("rec_texts") or res.get("rec_text")
        elif hasattr(res, "rec_texts"):
            texts = res.rec_texts
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            for t in texts:
                if t and str(t).strip():
                    lines.append(str(t).strip())
            continue

        # --- 2.x format: res is a list of [box, (text, conf)] ---
        if isinstance(res, list):
            for item in res:
                try:
                    t = item[1][0]
                except (IndexError, TypeError):
                    continue
                if t and str(t).strip():
                    lines.append(str(t).strip())

    return "\n".join(lines)


@functools.lru_cache(maxsize=2)
def _easyocr_reader(lang: str):
    import easyocr

    langs = ["ar", "en"] if lang == "ar" else ["en"]
    return easyocr.Reader(langs, gpu=False)


def _ocr_easyocr(img: Image.Image, lang: str) -> str:
    import numpy as np

    reader = _easyocr_reader(lang)
    # paragraph=False keeps one entry per detected line (preserves TOC line/page
    # structure); detail=0 returns plain strings in reading order.
    result = reader.readtext(np.array(img), detail=0, paragraph=False)
    return "\n".join(str(line).strip() for line in result if line and str(line).strip())


# Strict transcription prompt for cloud vision models. They must NOT structure
# or interpret — only read text exactly as printed, top to bottom.
_VISION_PROMPT = (
    "You are an OCR engine. Transcribe ALL text visible in this image of a book's "
    "table-of-contents page, exactly as printed, preserving line breaks and reading "
    "order (right-to-left for Arabic). Include every page number. Do NOT translate, "
    "summarize, reorder, explain, or add anything. Output ONLY the raw transcribed text."
)


def _vision_client(backend: str):
    from openai import OpenAI

    if backend == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("OCR_BACKEND=gemini requires GEMINI_API_KEY in .env")
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=key,
        ), os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if backend == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("OCR_BACKEND=groq requires GROQ_API_KEY in .env")
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=key,
        ), os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OCR_BACKEND=openai requires OPENAI_API_KEY in .env")
    return OpenAI(api_key=key), os.getenv("OPENAI_VISION_MODEL", "gpt-4o")


def _ocr_vision(img: Image.Image, backend: str) -> str:
    client, model = _vision_client(backend)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _img_to_data_url(img)}},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def _ocr_image(img: Image.Image, lang: str) -> str:
    """Run the configured OCR backend on a single image."""
    backend = os.getenv("OCR_BACKEND", "paddle").lower()
    if backend == "paddle":
        return _ocr_paddle(img, lang)
    if backend == "easyocr":
        return _ocr_easyocr(img, lang)
    if backend in ("gemini", "openai", "groq"):
        return _ocr_vision(img, backend)
    raise RuntimeError(f"Unknown OCR_BACKEND: {backend!r}")


def transcribe_pages(pdf_path: str, start_page: int, end_page: int, lang: str) -> list[dict]:
    """OCR a PDF page range; return [{'pdf_page': int, 'text': str}, ...] (1-based)."""
    images = render_pages(pdf_path, start_page, end_page)
    out: list[dict] = []
    for offset, img in enumerate(images):
        pdf_page = start_page + offset
        out.append({"pdf_page": pdf_page, "text": _ocr_image(img, lang).strip()})
    return out


def transcribe_toc(pdf_path: str, start_page: int, end_page: int, lang: str) -> str:
    """Render the TOC page range and return the full verbatim transcription."""
    pages = transcribe_pages(pdf_path, start_page, end_page, lang)
    return "\n\n".join(f"[page {p['pdf_page']}]\n{p['text']}".strip() for p in pages).strip()
