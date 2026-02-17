import re
import tiktoken
import fitz
import ollama
import chromadb
import pandas
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rank_bm25 import BM25Okapi

# ── CONFIG ──────────────────────────────────────────────────────────────
VISION_MODEL  = "minicpm-v:8b"
TEXT_MODEL    = "deepseek-r1:8b"
EMBED_MODEL   = "mxbai-embed-large"
CHROMA_PATH   = "./chroma_db"
OLLAMA_URL    = "http://localhost:11434"

MODEL_CONTEXT_WINDOW = 128_000
RESERVED_TOKENS      = 3_000
TOKEN_BUDGET         = MODEL_CONTEXT_WINDOW - RESERVED_TOKENS

MAX_PAGES_DIRECT = 30
PAGE_SCALE       = 2.0

MIN_IMAGE_WIDTH  = 150
MIN_IMAGE_HEIGHT = 150

TOP_K    = 6
FINAL_K  = 3
SHOW_THINKING = True

JUNK_LINES = [
    "here is the requested image",
    "this is the table",
]

# ── HELPERS ──────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def describe_image(image_bytes: bytes, doc_context: str = "") -> str:
    """Vision model only on actual embedded image objects — never full pages."""
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role":    "user",
            "content": f"""Describe this image in detail.

If it contains text: transcribe ALL of it exactly.
If it is a UI screenshot: describe every section, panel, and label visible.
If it is a chart: describe axes, values, and trends.
If it is a table rendered as image: extract all rows and columns.
Be thorough — this description is used for semantic search.""",
            "images": [image_bytes]
        }]
    )
    return response.message.content


# ── PAGE EXTRACTION (two-stage, no full-page render) ─────────────────────

def extract_page(page: fitz.Page, page_num: int, doc_context: str = "") -> dict:
    """
    Stage 1 — PyMuPDF native text + tables (free, instant, zero GPU).
    Stage 2 — Vision model only on embedded image objects (targeted, fast).
    """
    result = {
        "page_num": page_num + 1,
        "text":     "",
        "tables":   [],
        "images":   []
    }

    # Stage 1a: native text
    raw_lines   = page.get_text("text").strip().splitlines()
    clean_lines = [
        line for line in raw_lines
        if line.strip()
        and len(line.strip()) >= 3
        and not any(j in line.lower() for j in JUNK_LINES)
    ]
    native_text = "\n".join(clean_lines).strip()
    if native_text:
        result["text"] = native_text

    # Stage 1b: native tables
    try:
        junk_cols = [
            c for c in df.columns
            if any(j in str(c).lower() for j in JUNK_LINES)
        ]

        tables = page.find_tables()
        for table in tables:
            df = table.to_pandas()
            df = df.drop(columns=junk_cols, errors="ignore")
            if not df.empty:
                result["tables"].append(df.to_markdown(index=False))
    except Exception:
        pass

    # Stage 2: embedded images only
    doc        = page.parent
    image_list = page.get_images(full=True)
    seen       = set()

    for img_info in image_list:
        try:
            import hashlib
            xref        = img_info[0]
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            width       = base_image.get("width",  0)
            height      = base_image.get("height", 0)

            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash in seen:
                continue
            seen.add(img_hash)

            description = describe_image(image_bytes, doc_context)
            result["images"].append({
                "width":       width,
                "height":      height,
                "description": description
            })
        except Exception as e:
            print(f"     ⚠️  Image on page {page_num + 1}: {e}")

    return result


def format_page_content(page_data: dict) -> str:
    """Combine text, tables, and image descriptions into one clean string."""
    parts = [f"=== PAGE {page_data['page_num']} ==="]

    if page_data["text"]:
        parts.append(page_data["text"])

    for i, table in enumerate(page_data["tables"]):
        parts.append(f"\n[TABLE {i + 1}]\n{table}")

    for i, img in enumerate(page_data["images"]):
        parts.append(
            f"\n[IMAGE {i + 1} — {img['width']}x{img['height']}px]\n"
            f"{img['description']}"
        )

    return "\n\n".join(parts)


def extract_pdf(pdf_path: str) -> str:
    """Extract full PDF content using two-stage approach."""
    fitz_doc    = fitz.open(pdf_path)
    doc_context = fitz_doc[0].get_text("text")[:1000] if len(fitz_doc) > 0 else ""
    all_pages   = []

    print(f"  → Extracting {len(fitz_doc)} page(s)...")

    for page_num in range(len(fitz_doc)):
        page      = fitz_doc[page_num]
        page_data = extract_page(page, page_num, doc_context)
        content   = format_page_content(page_data)
        all_pages.append(content)

        has_text   = "✓ text"                                 if page_data["text"]   else "✗ text"
        has_tables = f"✓ {len(page_data['tables'])} table(s)" if page_data["tables"] else "✗ tables"
        has_images = f"✓ {len(page_data['images'])} image(s)" if page_data["images"] else "✗ images"
        print(f"     Page {page_num + 1}: {has_text} | {has_tables} | {has_images}")

    fitz_doc.close()
    return "\n\n".join(all_pages)


# ── DECISION LOGIC ────────────────────────────────────────────────────────

def should_use_rag(pdf_path: str) -> tuple[bool, dict]:
    """
    Decide Direct vs RAG based on:
      1. Page count hard ceiling (> MAX_PAGES_DIRECT → always RAG)
      2. Token count estimation via sampling
    """
    doc       = fitz.open(pdf_path)
    num_pages = len(doc)

    diagnostics = {
        "pdf":              pdf_path,
        "num_pages":        num_pages,
        "token_budget":     TOKEN_BUDGET,
        "decision_reason":  None,
        "estimated_tokens": None,
    }

    # Check 1: hard page ceiling
    if num_pages > MAX_PAGES_DIRECT:
        diagnostics["decision_reason"] = (
            f"Page count ({num_pages}) exceeds ceiling ({MAX_PAGES_DIRECT})"
        )
        doc.close()
        return True, diagnostics

    # Check 2: token estimation via sampling
    sample_indices = set([0, num_pages - 1])
    if num_pages > 2:
        step = max(1, num_pages // 4)
        for i in range(step, num_pages - 1, step):
            sample_indices.add(i)
            if len(sample_indices) >= 5:
                break

    sample_tokens = []
    print(f"  → Sampling {len(sample_indices)} of {num_pages} page(s) for token estimation...")

    for idx in sorted(sample_indices):
        try:
            page      = doc[idx]
            page_data = extract_page(page, idx)
            content   = format_page_content(page_data)
            tokens    = count_tokens(content)
            sample_tokens.append(tokens)
            print(f"     Page {idx + 1}: ~{tokens:,} tokens")
        except Exception as e:
            import traceback
            print(f"     ⚠️  Could not sample page {idx + 1}:")
            traceback.print_exc()

    doc.close()

    if not sample_tokens:
        diagnostics["decision_reason"] = "Sampling failed — defaulting to RAG"
        return True, diagnostics

    avg_tokens        = sum(sample_tokens) / len(sample_tokens)
    estimated_total   = int(avg_tokens * num_pages)
    diagnostics["estimated_tokens"] = estimated_total

    print(f"  → Avg tokens/page: ~{avg_tokens:,.0f}")
    print(f"  → Estimated total: ~{estimated_total:,} / {TOKEN_BUDGET:,} budget")

    if estimated_total > TOKEN_BUDGET:
        diagnostics["decision_reason"] = (
            f"Estimated tokens ({estimated_total:,}) exceeds budget ({TOKEN_BUDGET:,})"
        )
        return True, diagnostics

    diagnostics["decision_reason"] = (
        f"Fits in context: ~{estimated_total:,} tokens, {num_pages} pages"
    )
    return False, diagnostics


# ── DIRECT MODE ──────────────────────────────────────────────────────────

def answer_direct(pdf_path: str, question: str) -> str:
    """
    Small docs: extract everything natively, send full content to LLM.
    GPU only used briefly for any embedded images.
    """
    full_content = extract_pdf(pdf_path)

     # ── TEMPORARY DEBUG — remove after confirming image content ──
    print("\n=== EXTRACTED CONTENT ===")
    print(full_content)
    print("=== END CONTENT ===\n")
    # ─────────────────────────────────────────────────────────────

    total_tokens = count_tokens(full_content)
    print(f"  → Total: {total_tokens:,} tokens — sending to {TEXT_MODEL}...")


    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{
            "role": "user",
            "content": f"""You have been given the complete content of a PDF document.
Answer the question using ONLY this content.
If the answer is not present, say so clearly.

DOCUMENT CONTENT:
{full_content}

QUESTION: {question}

Answer:"""
        }]
    )

    content = response.message.content
    if SHOW_THINKING:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            print("\n=== DeepSeek-R1 Reasoning ===")
            print(match.group(1).strip()[:1000])

    return strip_thinking(content)


# ── RAG MODE ─────────────────────────────────────────────────────────────

def answer_rag(question: str) -> str:
    """
    Large docs: hybrid vector + BM25 search, then LLM on top chunks.
    Requires prior ingestion via ingest_pdf.py.
    """
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embeddings"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(
            name="pdf_rag",
            embedding_function=embedding_fn
        )
    except Exception:
        return (
            "⚠️  No indexed documents found.\n"
            "This document requires RAG but hasn't been ingested yet.\n"
            "Please run: python ingest_pdf.py"
        )

    if collection.count() == 0:
        return "⚠️  Vector store is empty. Run ingest_pdf.py first."

    # Hybrid search: vector + BM25
    results   = collection.query(
        query_texts=[question],
        n_results=min(TOP_K * 2, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    tokenized = [d.lower().split() for d in docs]
    bm25      = BM25Okapi(tokenized)
    bm25_raw  = bm25.get_scores(question.lower().split())
    bm25_max  = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_norm = [s / bm25_max for s in bm25_raw]

    combined = sorted([
        {
            "doc":   doc,
            "meta":  meta,
            "score": round(0.6 * (1 - dist) + 0.4 * bm25_norm[i], 3)
        }
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances))
    ], key=lambda x: x["score"], reverse=True)[:FINAL_K]

    print("\n=== Retrieved Sources ===")
    for i, r in enumerate(combined, 1):
        t    = r["meta"].get("type",   "?").upper()
        p    = r["meta"].get("page",   "?")
        s    = r["meta"].get("source", "?")
        sc   = r["score"]
        print(f"[{i}] {t} | page {p} | {s} | score={sc}")
        print(f"    {r['doc'][:200]}...")

    context = "\n\n---\n\n".join([
        f"[Page {r['meta'].get('page', '?')} | score={r['score']}]\n{r['doc']}"
        for r in combined
    ])

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using ONLY the context below.
If the answer is not present, say "I cannot find this in the documents."

Context:
{context}

Question: {question}
Answer:"""
        }]
    )
    return strip_thinking(response.message.content)


# ── SMART ROUTER ─────────────────────────────────────────────────────────

def smart_query(pdf_path: str, question: str) -> str:
    """
    Auto-selects strategy:
      Pages > 30   or   tokens > 125K  →  RAG (pre-indexed)
      Otherwise                        →  Direct (extract + send in one shot)
    """
    print(f"\n📋 Analysing: {pdf_path}")
    use_rag, diag = should_use_rag(pdf_path)

    print(f"\n{'='*52}")
    print(f"  Pages:            {diag['num_pages']}")
    if diag.get("estimated_tokens"):
        print(f"  Estimated tokens: {diag['estimated_tokens']:,}")
    print(f"  Token budget:     {diag['token_budget']:,}")
    print(f"  Decision:         {'RAG' if use_rag else 'Direct'}")
    print(f"  Reason:           {diag['decision_reason']}")
    print(f"{'='*52}\n")

    if use_rag:
        print("🔍 Using RAG pipeline...")
        return answer_rag(question)
    else:
        print("⚡ Using direct processing...")
        return answer_direct(pdf_path, question)


# ── ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf = "./pdfs/test.pdf"

    print("Smart PDF Q&A — auto-selects Direct vs RAG")
    print(f"Set SHOW_THINKING = True to see deepseek-r1 reasoning\n")
    print("Type 'quit' to exit\n")

    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            answer = smart_query(pdf, q)
            print(f"\n=== Answer ===\n{answer}\n")