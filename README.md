# PDF Scholar — Table of Contents Extractor

A Python service that reads a book's **Table of Contents** and returns a clean
hierarchy of **chapters → lessons**, each with its **start/end page**.

It works in **English and Arabic**, and it is built so the AI model **cannot
rewrite your book** — it may only fix obvious spelling/OCR typos. Every title is
verified against the original text.

## How it works

```
PDF + TOC page range
        │
        ▼
1. Render TOC pages → images            (PyMuPDF)
        │
        ▼
2. OCR each image → verbatim text       (PaddleOCR, local — pluggable)
        │
        ▼
3. Structure into chapters/lessons      (Cerebras gpt-oss-120b)
   + page ranges, EN/AR
   ❗ forbidden to edit — spelling only
        │
        ▼
4. Guard: verify every title exists      (fuzzy match vs. source)
   in the OCR text → title_verified + warnings
        │
        ▼
   JSON response
```

**Why images instead of raw PDF text?** Raw text extraction is unreliable for
Arabic (reversed letters, broken ligatures) and impossible for scanned books.
Rendering to an image + OCR gives uniform, layout-aware reading. The OCR step is
the only part that "reads"; the AI only *organizes* what was read.

## Setup

```bash
pip install -r requirements.txt
```

`.env`:

```ini
CEREBRAS_API_KEY=...        # required (structuring step)
CEREBRAS_MODEL=gpt-oss-120b
OCR_BACKEND=paddle          # paddle | easyocr | gemini | openai
RENDER_DPI=200
# GEMINI_API_KEY=...        # only if OCR_BACKEND=gemini
# OPENAI_API_KEY=...        # only if OCR_BACKEND=openai
```

### OCR backends

| `OCR_BACKEND` | Type | API key | Notes |
|---------------|------|---------|-------|
| `easyocr` *(default on Windows)* | local | none | `pip install easyocr`. Reliable on Windows CPU. Weaker on page-number digits. |
| `paddle` | local | none | ⚠️ **Broken on Windows** — paddlepaddle 3.3.1 oneDNN/PIR executor crashes on real-sized pages (`ConvertPirAttribute2RuntimeAttribute`). Avoid here. |
| `gemini` | cloud | `GEMINI_API_KEY` | Most accurate on page numbers / Arabic; free tier available. Best if local OCR drops page numbers. |
| `openai` | cloud | `OPENAI_API_KEY` | GPT-4o vision |

> **Page-number accuracy:** local OCR (easyocr) reads titles well but can miss
> the page number on dotted-leader lines. If your output has `null` page
> numbers, switch to `OCR_BACKEND=gemini` for the most reliable digits.

> **Note:** Cerebras models are text-only and **cannot read images**, so the OCR
> step always runs on one of the backends above. Cerebras only does step 3.

## Run

```bash
uvicorn main:app --reload
```

## API

### `POST /extract-toc`

Form fields:

| Field | Required | Description |
|-------|----------|-------------|
| `file` | one of | The PDF, uploaded as multipart |
| `pdf_path` | one of | …or a path to a PDF already on the server |
| `toc_start_page` | yes | First TOC page (1-based, inclusive) |
| `toc_end_page` | yes | Last TOC page (1-based, inclusive) |
| `lang` | no | `en` (default) or `ar` |

Example:

```bash
curl -X POST http://localhost:8000/extract-toc \
  -F "file=@book.pdf" \
  -F "toc_start_page=3" \
  -F "toc_end_page=5" \
  -F "lang=ar"
```

Response:

```json
{
  "language": "ar",
  "toc_start_page": 3,
  "toc_end_page": 5,
  "source_transcription": "…full OCR text of the TOC pages…",
  "chapters": [
    {
      "title": "الفصل الأول: المادة وخصائصها",
      "start_page": 12,
      "end_page": 45,
      "title_verified": true,
      "lessons": [
        { "title": "الدرس الأول: حالات المادة", "start_page": 12, "end_page": 18, "title_verified": true },
        { "title": "الدرس الثاني: التغيرات الفيزيائية", "start_page": 19, "end_page": 45, "title_verified": true }
      ]
    }
  ],
  "warnings": []
}
```

- `title_verified: false` and a matching entry in `warnings` mean the AI returned
  a title that does not appear in the source — i.e. it may have altered content.
  Treat those entries with suspicion.
- Missing `end_page` values are derived from the next item's `start_page`.

### `GET /health`

Returns the active OCR backend and Cerebras model.
