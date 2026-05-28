import asyncio
import json
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

jobs: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/scan")
async def scan_folder(folder: str = Form(...)):
    folder = folder.strip().strip('"').strip("'")
    path = Path(folder)
    if not path.exists():
        return JSONResponse({"error": f"Path does not exist: {path}"}, status_code=400)
    if not path.is_dir():
        return JSONResponse({"error": f"Not a folder: {path}"}, status_code=400)
    pdfs = [str(p) for p in sorted(path.rglob("*.pdf"))]
    if not pdfs:
        return JSONResponse({"error": f"No PDF files found in: {path}"}, status_code=400)
    return {"pdfs": pdfs}


@app.post("/generate")
async def generate(pdf_path: str = Form(...), lang: str = Form("auto"), pages: int = Form(30)):
    if not Path(pdf_path).exists():
        return JSONResponse({"error": "PDF file not found."}, status_code=400)

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "extracting",
        "progress": 0,
        "current": "Extracting PDF...",
        "done_sections": 0,
        "total_sections": 0,
        "language": lang,
        "results": [],
        "error": None,
        "book_filename": Path(pdf_path).stem,
    }
    asyncio.create_task(run_pipeline(job_id, pdf_path, lang, pages))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found."}, status_code=404)
    return jobs[job_id]


@app.get("/download/{job_id}.json")
async def download_json(job_id: str):
    out = OUTPUT_DIR / f"{job_id}.json"
    if not out.exists():
        return JSONResponse({"error": "Results not ready."}, status_code=404)
    return FileResponse(out, filename="study_data.json", media_type="application/json")


@app.get("/download/{job_id}.md")
async def download_md(job_id: str):
    out = OUTPUT_DIR / f"{job_id}.md"
    if not out.exists():
        return JSONResponse({"error": "Results not ready."}, status_code=404)
    return FileResponse(out, filename="study_notes.md", media_type="text/markdown")


def section_to_markdown(s: dict) -> str:
    lang = s.get("language", "en")
    ar = lang == "ar"

    lines = [f"## 📚 {s.get('section_number', '?')}: {s.get('title', '')}"]
    lines.append("")
    lines.append(f"**{'الموضوع' if ar else 'Subject'}:** {s.get('subject', 'general')} | "
                 f"**{'وقت القراءة' if ar else 'Reading time'}:** {s.get('estimated_reading_minutes', 0)} {'دقائق' if ar else 'min'}")
    lines.append("")

    if s.get("learning_objectives"):
        lines.append(f"### 🎯 {'أهداف التعلم' if ar else 'Learning Objectives'}")
        for obj in s["learning_objectives"]:
            lines.append(f"- {obj}")
        lines.append("")

    if s.get("lesson_explanation"):
        lines.append(f"### 📖 {'شرح الدرس' if ar else 'Lesson Explanation'}")
        lines.append(s["lesson_explanation"])
        lines.append("")

    if s.get("key_terms"):
        lines.append(f"### 🔑 {'المصطلحات الرئيسية' if ar else 'Key Terms'}")
        for term in s["key_terms"]:
            lines.append(f"- **{term.get('term', '')}**: {term.get('definition', '')}")
        lines.append("")

    val = s.get("scientific_validation", {})
    if val.get("notes"):
        lines.append(f"### ✅ {'التحقق العلمي' if ar else 'Scientific Validation'}")
        lines.append(val["notes"])
        lines.append("")

    exam = s.get("exam", {})
    if exam.get("multiple_choice") or exam.get("short_answer") or exam.get("problem_solving"):
        lines.append(f"### ❓ {'أسئلة الامتحان' if ar else 'Exam Q&A'}")
        for i, q in enumerate(exam.get("multiple_choice", []), 1):
            lines.append(f"**{'س' if ar else 'Q'}{i}.** {q.get('question', '')}")
            for opt in q.get("options", []):
                lines.append(f"  {opt}")
            lines.append(f"**{'الإجابة' if ar else 'Answer'}:** {q.get('answer', '')} — {q.get('explanation', '')}")
            lines.append("")
        for i, q in enumerate(exam.get("short_answer", []), 1):
            lines.append(f"**{'س قصير' if ar else 'SA'}{i}.** {q.get('question', '')}")
            lines.append(f"**{'الإجابة' if ar else 'Answer'}:** {q.get('answer', '')}")
            lines.append("")
        for i, q in enumerate(exam.get("problem_solving", []), 1):
            lines.append(f"**{'مسألة' if ar else 'Problem'} {i}.** {q.get('problem', '')}")
            lines.append(f"**{'الحل' if ar else 'Solution'}:** {q.get('solution', '')}")
            lines.append("")

    return "\n".join(lines)


async def run_pipeline(job_id: str, pdf_path: str, lang_choice: str = "auto", max_pages: int = 30):
    try:
        chunks, detected_lang = await asyncio.to_thread(extract_pdf, pdf_path, job_id, max_pages)
        book_lang = lang_choice if lang_choice in ("ar", "en") else detected_lang
        total = len(chunks)
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["total_sections"] = total
        jobs[job_id]["language"] = book_lang
        jobs[job_id]["current"] = f"Processing section 0 of {total}..."

        all_results = []
        for i, chunk in enumerate(chunks):
            jobs[job_id]["current"] = f"Processing section {i + 1} of {total}..."
            jobs[job_id]["progress"] = int((i / total) * 100)
            jobs[job_id]["done_sections"] = i

            result = await asyncio.to_thread(process_chunk, chunk, i + 1, total, book_lang)
            all_results.append(result)
            jobs[job_id]["results"] = all_results

        # Save JSON
        json_path = OUTPUT_DIR / f"{job_id}.json"
        json_payload = {
            "book_filename": jobs[job_id]["book_filename"],
            "language": book_lang,
            "total_sections": total,
            "sections": all_results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

        # Save MD
        md_path = OUTPUT_DIR / f"{job_id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {jobs[job_id]['book_filename']}\n\n")
            for s in all_results:
                f.write(section_to_markdown(s) + "\n\n---\n\n")

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
