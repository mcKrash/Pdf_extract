# 📚 PDF Scholar

An AI-powered tool that turns any PDF book into structured study material — lesson breakdowns, key terms, scientific validation, and exam Q&A — fully formatted in Arabic or English.

## ✨ Features

- 📄 **PDF text extraction** with Arabic Presentation Forms normalization (handles legacy PDF encoding)
- 🌍 **Bilingual support** — Arabic & English (auto-detect or force language)
- 🧠 **AI-generated study material** per chapter/section:
  - 🎯 Learning Objectives
  - 📖 Full Lesson Explanation
  - 🔑 Key Terms & Definitions
  - ✅ Scientific Validation (formulas, equations)
  - ❓ Exam Q&A (multiple choice, short answer, problem solving)
- 🚀 **Powered by Cerebras** — ultra-fast inference (1,800+ tokens/sec)
- 🛡️ **Language enforcement** — guarantees output matches selected language (auto-translation fallback)
- 🧹 **Smart filtering** — skips TOC, copyright, and corrupted pages automatically
- 💾 **Downloadable Markdown** of all generated content

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

Get a free key at [cloud.cerebras.ai](https://cloud.cerebras.ai) (1M tokens/day free).

### 3. Run

```bash
python main.py
```

Open `http://localhost:8000` in your browser.

## 🧭 Usage

1. **Enter folder path** containing your PDF books (e.g., `C:\Books`)
2. Click **Scan Folder** — picks up all PDFs
3. Select the PDF you want to process
4. Choose **output language** (Auto / Arabic / English)
5. Choose **pages to process** (default 30)
6. Click **Generate Study Material**
7. Watch live progress section-by-section
8. **Download** the full `.md` file when done

## 🏗️ Architecture

```
pdf-scholar/
├── main.py              # FastAPI server + job orchestration
├── extractor.py         # PyMuPDF text extraction + quality filtering
├── processor.py         # Cerebras AI calls + language enforcement
├── templates/
│   └── index.html       # Dark-mode UI with live progress
├── output/              # Generated .md files
└── requirements.txt
```

## ⚙️ Tech Stack

- **Backend**: Python 3.10+ · FastAPI · Uvicorn
- **PDF Extraction**: PyMuPDF (fitz)
- **AI Inference**: Cerebras Cloud API (gpt-oss-120b)
- **Frontend**: Vanilla HTML + JavaScript (dark mode)

## 🎯 Language Enforcement

When Arabic output is selected, the tool guarantees Arabic-only output through a 3-layer safety net:

1. **Strict prompt** — system & user messages explicitly demand Arabic
2. **Auto-retry** — if response contains English, retries with stricter warning
3. **Translation fallback** — final pass translates any English remnants into polished Arabic
4. **Last-resort block** — if all fails, returns an Arabic placeholder (never English)

## 📊 Performance

| Operation | Speed |
|-----------|-------|
| PDF extraction (PyMuPDF) | ~5 seconds for 100 pages |
| AI processing (Cerebras) | ~1 second per section |
| 30-page book end-to-end | ~1–2 minutes |

## 🤝 Contributing

PRs welcome. Open an issue first to discuss bigger changes.

## 📜 License

MIT
