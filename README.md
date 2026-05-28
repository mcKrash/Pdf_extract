# 📚 PDF Scholar

Turn any PDF textbook into structured, exam-ready study material with AI — full lessons, key terms, scientific validation, and exam Q&A. Outputs both human-readable Markdown and DB-ready JSON.

## ✨ Features

### 🎯 Smart Content Extraction
- 📄 **PDF text extraction** with Arabic Presentation Forms normalization (handles legacy PDF encoding)
- 🧹 **Auto-skips intro pages** — finds first `Chapter 1` / `الفصل الأول` / `Unit 1` and starts there
- 🚫 **Filters junk pages** — covers, copyright, ISBN, table of contents
- 🔍 **Quality filter** — drops chunks with insufficient real text

### 🧠 AI-Generated Study Material
Each section produces a complete structured lesson:
- 🎯 **Learning Objectives** — 3 to 5 specific outcomes
- 📖 **Full Lesson Explanation** — detailed, exam-grade content
- 🔑 **Key Terms & Definitions** — dictionary-style entries
- ✅ **Scientific Validation** — formula/equation correctness check
- ❓ **Exam Q&A** — 5 multiple choice, 3 short answer, 1–2 problem-solving

### 🌍 Bilingual & Reliable
- Arabic & English support (auto-detect or force language)
- **3-layer language guarantee** — strict prompt → translation pass → fallback
- **2-attempt content guarantee** — if model gives up, retry with keyword-forced prompt
- Result: every section produces real, usable exam content (no "text not analyzable" outputs)

### 🎨 Card Grid UI
- Subject-themed cards (chemistry 🧪 green · physics 🔬 blue · math 📐 amber · biology 🧬 pink)
- Stats per card: objectives count, terms count, total questions, validity bar
- Click any card → full-screen modal with the complete lesson
- **RTL layout** auto-switches for Arabic content
- Glass-morphism dark theme

### 💾 Two Output Formats
- **JSON** — DB-ready rows for integration into a learning app
- **Markdown** — clean human-readable study notes

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Cerebras API key

Create a `.env` file:

```bash
CEREBRAS_API_KEY=your_cerebras_key_here
```

Get a free key at [cloud.cerebras.ai](https://cloud.cerebras.ai) — 1M tokens/day free.

### 3. Run

```bash
python main.py
```

Open `http://localhost:8000` in your browser.

## 🧭 Usage

1. **Enter folder path** containing your PDF books (e.g., `C:\Books`)
2. Click **Scan Folder** — finds all PDFs in that folder
3. **Select PDF** from the dropdown
4. Choose **Output Language** (Auto / Arabic / English)
5. Choose **Pages to Process** (default 30)
6. Click **✨ Generate Study Material**
7. Watch sections appear live as they're processed
8. **Click any card** to see the full lesson
9. **Download JSON or Markdown** when done

## 📦 JSON Schema (per section)

```json
{
  "section_number": 1,
  "language": "ar",
  "title": "هاليدات الألكيل",
  "subject": "chemistry",
  "estimated_reading_minutes": 8,
  "learning_objectives": [
    "التعرف على الصيغة العامة لهاليدات الألكيل",
    "تمييز الأنواع الثلاثة (أولي، ثانوي، ثالثي)"
  ],
  "lesson_explanation": "هاليدات الألكيل هي مركبات عضوية...",
  "key_terms": [
    { "term": "هاليد ألكيل", "definition": "مركب عضوي..." }
  ],
  "scientific_validation": {
    "is_valid": true,
    "notes": "الصيغة العامة CₙH₂ₙ₊₁X صحيحة لجميع هاليدات الألكيل"
  },
  "exam": {
    "multiple_choice": [
      {
        "question": "أي من المركبات التالية هاليد ألكيل أولي؟",
        "options": ["أ) CH₃CH₂Br", "ب) ...", "ج) ...", "د) ..."],
        "answer": "أ",
        "explanation": "ذرة البروم مرتبطة بذرة كربون أولية"
      }
    ],
    "short_answer": [
      { "question": "...", "answer": "..." }
    ],
    "problem_solving": [
      { "problem": "...", "solution": "..." }
    ]
  }
}
```

The top-level JSON also contains `book_filename`, `language`, `total_sections`, and `sections[]`.

## 🏗️ Architecture

```
pdf-scholar/
├── main.py              # FastAPI server + job orchestration
├── extractor.py         # PyMuPDF text extraction + intro-skip + quality filter
├── processor.py         # Cerebras AI calls + JSON enforcement + content guarantee
├── templates/
│   └── index.html       # Card-grid UI with subject theming
├── output/              # Generated .json and .md files
└── requirements.txt
```

## ⚙️ Tech Stack

- **Backend**: Python 3.10+ · FastAPI · Uvicorn
- **PDF Extraction**: PyMuPDF (fitz) with NFKC normalization
- **AI Inference**: Cerebras Cloud API (`gpt-oss-120b`) — 1,800 tokens/sec
- **Frontend**: Vanilla HTML + JavaScript (no framework, dark mode)

## 🛡️ Guarantees

| Concern | How it's handled |
|---------|-----------------|
| English in Arabic books | Strict prompt → retry → auto-translation → forced fallback |
| Garbled OCR text | Keyword extraction + domain-knowledge generation |
| Empty/refused sections | "Empty response" detector triggers forced 2nd attempt |
| Intro/cover pages | Auto-skip until first chapter marker |
| TOC / copyright pages | Pattern-based skip filter |

## 📊 Performance

| Operation | Speed |
|-----------|-------|
| PDF extraction | ~5 seconds for 100 pages |
| AI processing | ~2–3 seconds per section |
| 30-page book end-to-end | ~1–2 minutes |

## 📜 License

MIT
