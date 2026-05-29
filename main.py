"""
PDF Scholar — Table-of-Contents extraction service.

Pipeline:
    1. Render the given TOC page range to images          (ocr.render_pages)
    2. OCR each image -> verbatim transcription           (ocr.transcribe_toc)
    3. Cerebras structures it: chapters -> lessons + pages (structure.structure_toc)
    4. Guard: verify every title exists in the source     (verify.verify_titles)

The AI is forbidden from editing content — spelling fixes only — and step 4
flags anything it may have altered.
"""

import asyncio
import uuid
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse

from ocr import transcribe_toc
from structure import structure_toc
from verify import verify_titles
from unit import extract_unit

load_dotenv()

app = FastAPI(title="PDF Scholar — TOC Extractor")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


def _run_pipeline(pdf_path: str, start_page: int, end_page: int, lang: str) -> dict:
    """Blocking pipeline (runs in a worker thread)."""
    transcription = transcribe_toc(pdf_path, start_page, end_page, lang)
    if not transcription.strip():
        raise ValueError(
            "OCR produced no text for the given page range. "
            "Check toc_start_page / toc_end_page, or try a different OCR_BACKEND."
        )
    chapters = structure_toc(transcription, lang)
    warnings = verify_titles(chapters, transcription)
    return {
        "language": lang,
        "toc_start_page": start_page,
        "toc_end_page": end_page,
        "source_transcription": transcription,
        "chapters": chapters,
        "warnings": warnings,
    }


@app.post("/extract-toc")
async def extract_toc(
    file: UploadFile | None = File(default=None),
    pdf_path: str = Form(default=""),
    toc_start_page: int = Form(...),
    toc_end_page: int = Form(...),
    lang: str = Form("en"),
):
    """Extract chapters & lessons from a PDF's table of contents.

    Provide EITHER an uploaded `file` OR a server-side `pdf_path`.
    `lang` must be "en" or "ar". Page numbers are 1-based and inclusive.
    """
    lang = (lang or "en").lower()
    if lang not in ("en", "ar"):
        return JSONResponse({"error": "lang must be 'en' or 'ar'."}, status_code=400)

    tmp_path: Path | None = None
    try:
        if file is not None:
            tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
            target = str(tmp_path)
        elif pdf_path.strip():
            target = pdf_path.strip().strip('"').strip("'")
            if not Path(target).exists():
                return JSONResponse({"error": f"PDF not found: {target}"}, status_code=400)
        else:
            return JSONResponse(
                {"error": "Provide either an uploaded 'file' or a 'pdf_path'."},
                status_code=400,
            )

        result = await asyncio.to_thread(
            _run_pipeline, target, toc_start_page, toc_end_page, lang
        )
        return result

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Extraction failed: {e}"}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/extract-unit")
async def extract_unit_endpoint(
    file: UploadFile | None = File(default=None),
    pdf_path: str = Form(default=""),
    pdf_start_page: int = Form(...),
    pdf_end_page: int = Form(...),
    lang: str = Form("ar"),
    question_types: str = Form("mcq,short,essay,problem"),
    questions_per_lesson: int = Form(6),
):
    """Extract a full unit: lessons + verbatim content + generated questions.

    Give the unit's range as PDF page numbers (what you see in the PDF viewer).
    `question_types`: comma-separated from mcq,short,essay,problem.
    """
    lang = (lang or "ar").lower()
    if lang not in ("en", "ar"):
        return JSONResponse({"error": "lang must be 'en' or 'ar'."}, status_code=400)
    q_types = [t.strip().lower() for t in question_types.split(",") if t.strip()]

    tmp_path: Path | None = None
    try:
        if file is not None:
            tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
            target = str(tmp_path)
        elif pdf_path.strip():
            target = pdf_path.strip().strip('"').strip("'")
            if not Path(target).exists():
                return JSONResponse({"error": f"PDF not found: {target}"}, status_code=400)
        else:
            return JSONResponse({"error": "Provide either an uploaded 'file' or a 'pdf_path'."},
                                status_code=400)

        result = await asyncio.to_thread(
            extract_unit, target, pdf_start_page, pdf_end_page, lang, q_types, questions_per_lesson
        )
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Unit extraction failed: {e}"}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/health")
async def health():
    import os
    return {
        "status": "ok",
        "ocr_backend": os.getenv("OCR_BACKEND", "paddle"),
        "cerebras_model": os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
