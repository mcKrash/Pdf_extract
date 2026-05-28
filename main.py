import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from extractor import extract_pdf
from processor import process_chunk

load_dotenv()

app = FastAPI()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store
jobs: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/scan")
async def scan_folder(folder: str = Form(...)):
    """Return list of PDF files found in the given folder."""
    folder = folder.strip().strip('"').strip("'")
    path = Path(folder)
    if not path.exists():
        return JSONResponse({"error": f"Path does not exist: {path.resolve()}"}, status_code=400)
    if not path.is_dir():
        return JSONResponse({"error": f"Not a folder: {path.resolve()}"}, status_code=400)
    pdfs = [str(p) for p in sorted(path.rglob("*.pdf"))]
    if not pdfs:
        return JSONResponse({"error": f"No PDF files found in: {path.resolve()}"}, status_code=400)
    return {"pdfs": pdfs}


@app.post("/generate")
async def generate(pdf_path: str = Form(...), lang: str = Form("auto"), pages: int = Form(30)):
    """Start the extraction + AI processing pipeline."""
    if not Path(pdf_path).exists():
        return JSONResponse({"error": "PDF file not found."}, status_code=400)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "extracting",
        "progress": 0,
        "current": "Extracting PDF...",
        "done_sections": 0,
        "total_sections": 0,
        "results": [],
        "error": None,
    }

    asyncio.create_task(run_pipeline(job_id, pdf_path, lang, pages))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return jobs[job_id]


@app.get("/download/{job_id}")
async def download(job_id: str):
    out = OUTPUT_DIR / f"{job_id}.md"
    if not out.exists():
        return JSONResponse({"error": "Results not ready."}, status_code=404)
    return FileResponse(out, filename="study_notes.md", media_type="text/markdown")


async def run_pipeline(job_id: str, pdf_path: str, lang_choice: str = "auto", max_pages: int = 30):
    try:
        # Step 1: Extract
        chunks, detected_lang = await asyncio.to_thread(extract_pdf, pdf_path, job_id, max_pages)
        book_lang = lang_choice if lang_choice in ("ar", "en") else detected_lang
        total = len(chunks)
        jobs[job_id]["language"] = book_lang
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["total_sections"] = total
        jobs[job_id]["current"] = f"Processing section 0 of {total}..."

        all_results = []

        # Step 2: Process each chunk
        for i, chunk in enumerate(chunks):
            jobs[job_id]["current"] = f"Processing section {i + 1} of {total}..."
            jobs[job_id]["progress"] = int((i / total) * 100)
            jobs[job_id]["done_sections"] = i

            result = await asyncio.to_thread(process_chunk, chunk, i + 1, total, book_lang)
            all_results.append(result)
            jobs[job_id]["results"] = all_results

        # Step 3: Save
        out_path = OUTPUT_DIR / f"{job_id}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(r + "\n\n---\n\n")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["done_sections"] = total
        jobs[job_id]["current"] = "Done!"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
