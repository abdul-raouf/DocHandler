# 📄 DocHandler

A fully local, privacy-preserving PDF question-answering system that combines 
OCR, vision models, and LLM reasoning — no API keys, no data leaving your machine.

---

## 🧠 How It Works

SmartPDF Chat automatically selects the best strategy based on your document size:

- **Direct Mode** — small documents are extracted once, cached for the session, 
  and sent in full to the LLM. Fast, accurate, no retrieval errors.
- **RAG Mode** — large documents are chunked, embedded into a vector store, 
  and retrieved with hybrid search (vector + BM25) before LLM reasoning.

Every session maintains **conversation history** so follow-up questions like 
*"name them"* or *"what were they reported for?"* resolve correctly without 
re-processing the document.

---

## ✨ Features

- **Multimodal extraction** — handles text, tables, and embedded images in one pipeline
- **Vision OCR** — uses `minicpm-v` to read screenshots, UI captures, charts, and Arabic text
- **Smart routing** — auto-selects Direct vs RAG based on page count and token estimation
- **Session cache** — document extracted once per session, reused across all questions
- **Conversation memory** — proper chat history with automatic summarisation of older turns
- **Hybrid search** — combines vector similarity and BM25 keyword scoring for better retrieval
- **100% local** — all models run via Ollama, nothing sent to external APIs

---

## 🏗️ Architecture
```
PDF Input
   │
   ├── PyMuPDF ──────────────► Native text + tables   (instant, zero GPU)
   └── minicpm-v ────────────► Embedded image OCR     (targeted, fast)
                │
                ▼
         Session Cache
         (extracted once)
                │
      ┌─────────┴──────────┐
      │                    │
  Direct Mode           RAG Mode
  (< 30 pages,          (> 30 pages or
   fits in context)      > 125K tokens)
      │                    │
      │              ChromaDB + BM25
      │              Hybrid Search
      └─────────┬──────────┘
                │
         deepseek-r1:8b
         (with conversation
              history)
                │
            Answer
```

---

## 🛠️ Models Used

| Role | Model | Why |
|---|---|---|
| Text reasoning + Q&A | `deepseek-r1:8b` | Strong reasoning, 128K context, chain-of-thought |
| Image / OCR | `minicpm-v:8b` | Best local vision model for screenshots and Arabic text |
| Embeddings | `mxbai-embed-large` | Outperforms OpenAI `text-embedding-3-small` locally |

---

## 📦 Installation

**1. Install Ollama and pull models**
```bash
curl -fsSL https://ollama.com/install.sh | sh   # Linux/Mac
# Windows: download from https://ollama.com

ollama pull deepseek-r1:8b
ollama pull minicpm-v:8b
ollama pull mxbai-embed-large
```

**2. Clone and install dependencies**
```bash
git clone https://github.com/yourusername/smartpdf-chat
cd smartpdf-chat

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pymupdf pandas tabulate tiktoken rank_bm25 \
            chromadb langchain langchain-community \
            langchain-ollama ollama python-dotenv docling
```

**3. Add your PDFs**
```bash
mkdir pdfs
# copy your PDF files into the pdfs/ folder
```

---

## 🚀 Usage

### Interactive chat (recommended)
```bash
python query3.py                        # uses ./pdfs/test.pdf by default
python query3.py ./pdfs/your_file.pdf  # specify a PDF
```

### Pre-index large documents (RAG mode)
```bash
python ingest_pdf.py    # indexes all PDFs in ./pdfs/ into ChromaDB
```

### Session commands
| Command | Action |
|---|---|
| `history` | Show conversation so far |
| `reset` | Start a fresh session |
| `quit` | Exit |

---

## 📁 Project Structure
```
smartpdf-chat/
├── pdfs/               # Place your PDF files here
├── chroma_db/          # Auto-created vector store (RAG mode)
├── ingest_pdf.py       # Pre-index PDFs for RAG mode
├── query3.py           # Main interactive chat interface
└── README.md
```

---

## ⚙️ Configuration

All settings are at the top of each file:
```python
# Models
VISION_MODEL  = "minicpm-v:8b"       # swap for gemma3:12b, llava:13b etc.
TEXT_MODEL    = "deepseek-r1:8b"     # swap for any Ollama text model
EMBED_MODEL   = "mxbai-embed-large"  # swap for nomic-embed-text etc.

# Routing thresholds
MAX_PAGES_DIRECT  = 30               # pages above this always use RAG
TOKEN_BUDGET      = 125_000          # token ceiling for direct mode

# RAG settings
CHUNK_SIZE    = 400                  # characters per chunk
CHUNK_OVERLAP = 80                   # overlap between chunks
TOP_K         = 6                    # candidates fetched before reranking
FINAL_K       = 3                    # chunks passed to LLM

# Session settings
MAX_FULL_TURNS      = 6              # recent turns kept in full
HISTORY_TOKEN_LIMIT = 4_000          # token limit before summarisation
```

---

## 🔄 How Direct vs RAG Is Chosen
```
Document received
       │
       ├─ Pages > 30?          ──► RAG  (too slow to OCR at query time)
       │
       ├─ Sample 3-5 pages
       │  Estimate total tokens
       │
       ├─ Tokens > 125K?       ──► RAG  (exceeds model context window)
       │
       └─ All clear            ──► Direct (extract once, cache, answer)
```

---

## 🔒 Privacy

All processing is fully local:
- **Ollama** serves all models on `localhost:11434`
- **ChromaDB** stores embeddings on disk in `./chroma_db/`
- No internet connection required after model download
- No telemetry, no API calls, no data leaves your machine

---

## 🗺️ Roadmap

- [ ] Streamlit / Gradio web UI
- [ ] Multi-PDF sessions
- [ ] RAGAs evaluation framework
- [ ] Reranking with FlashRank
- [ ] Semantic chunking
- [ ] Export conversation history

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com) — local model serving
- [Docling](https://github.com/DS4SD/docling) — PDF structure extraction  
- [ChromaDB](https://www.trychroma.com) — vector store
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF rendering and text extraction
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) — vision model
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — reasoning model