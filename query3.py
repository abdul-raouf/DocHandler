"""
Current problem with query2.py: PDF text extraction + image OCR + token counting is repeated every time we ask a question, even if the PDF doesn't change. 
This adds ~7 seconds to every question, which is a bad user experience.

Every question currently:
  PDF extraction     (~2 sec)   ← repeated every question
  Image OCR          (~5 sec)   ← repeated every question
  Token counting     (~0.1 sec) ← repeated every question
  LLM call           (~10 sec)  ← necessary, but context re-built every time

What we want:
  First question:
    PDF extraction   (~2 sec)   ← once
    Image OCR        (~5 sec)   ← once
    Token counting   (~0.1 sec) ← once
    LLM call         (~10 sec)  ← necessary

  Every subsequent question:
    LLM call         (~10 sec)  ← only this
"""


#Solution to this problem:
"""
Layer 1 — Document Cache
  Store extracted PDF content in memory for the session
  Avoids re-running PyMuPDF + vision model on every question

Layer 2 — Conversation History
  Store Q&A pairs and pass them to the LLM each turn
  Gives the model memory of previous questions
  (exactly how ChatGPT works)

Layer 3 — Conversation Summarisation
  When history gets too long, summarise older turns
  Keeps token count manageable over long sessions
  
  """


# ---------------File Start-----------------------------------------------------------------
import re
import time
import tiktoken
import fitz
import ollama
import chromadb
import hashlib
from dataclasses import dataclass, field
from typing import Optional
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

MAX_PAGES_DIRECT     = 30
MIN_IMAGE_WIDTH      = 150
MIN_IMAGE_HEIGHT     = 150

TOP_K                = 6
FINAL_K              = 3

# How many recent Q&A turns to keep in full before summarising older ones
MAX_FULL_TURNS       = 6

# When history token count exceeds this, summarise the oldest turns
HISTORY_TOKEN_LIMIT  = 4_000

JUNK_LINES = [
    "here is the requested image",
    "this is the table",
    "<!-- image -->",
]

SHOW_THINKING = False


# ── DATA STRUCTURES ──────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single question/answer exchange."""
    question: str
    answer:   str
    turn_num: int


@dataclass
class DocumentCache:
    """
    Cached extraction result for a PDF.
    Avoids re-running PyMuPDF + vision model on every question.
    """
    pdf_path:     str
    content:      str          # full extracted text
    token_count:  int
    use_rag:      bool
    extracted_at: float = field(default_factory=time.time)


@dataclass
class Session:
    """
    Holds everything for one conversation:
    - The cached document extraction
    - The full conversation history
    - A running summary of older turns (once history gets long)
    """
    pdf_path:       str
    doc_cache:      Optional[DocumentCache] = None
    turns:          list[Turn]              = field(default_factory=list)
    history_summary: str                   = ""   # summarised older turns
    total_questions: int                   = 0


# ── HELPERS ──────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def describe_image(image_bytes: bytes) -> str:
    """Vision model on embedded images only — no full page rendering."""
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role":    "user",
            "content": """Extract all text from this image exactly as it appears.
Preserve all content including any non-English text (Arabic, etc.).
Format output as markdown.
If you see sections or panels, use headers to separate them.
If you see a table, format it as a markdown table.
Do not describe or summarize — only transcribe what is written.""",
            "images": [image_bytes]
        }]
    )
    return response.message.content


# ── PAGE EXTRACTION ───────────────────────────────────────────────────────

def extract_page(page: fitz.Page, page_num: int) -> dict:
    result = {
        "page_num": page_num + 1,
        "text":     "",
        "tables":   [],
        "images":   []
    }

    # Stage 1a: native text — filter junk lines
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
        tables = page.find_tables()
        for table in tables:
            df = table.to_pandas()
            if df.empty:
                continue
            df.columns = [str(c) for c in df.columns]
            junk_cols  = [
                c for c in df.columns
                if any(j in str(c).lower() for j in JUNK_LINES)
            ]
            df = df.drop(columns=junk_cols, errors="ignore")
            if not df.empty:
                result["tables"].append(df.to_markdown(index=False))
    except Exception:
        pass

    # Stage 2: vision model only on embedded images
    doc        = page.parent
    image_list = page.get_images(full=True)
    seen       = set()

    for img_info in image_list:
        try:
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

            description = describe_image(image_bytes)
            result["images"].append({
                "width":       width,
                "height":      height,
                "description": description
            })
        except Exception as e:
            print(f"       Image on page {page_num + 1}: {e}")

    return result


def format_page_content(page_data: dict) -> str:
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
    """Extract full PDF content — runs once per session, cached after."""
    fitz_doc  = fitz.open(pdf_path)
    all_pages = []
    print(f"  → Extracting {len(fitz_doc)} page(s)...")

    for page_num in range(len(fitz_doc)):
        page      = fitz_doc[page_num]
        page_data = extract_page(page, page_num)
        content   = format_page_content(page_data)
        all_pages.append(content)

        has_text   = "✓ text"   if page_data["text"]   else "✗ text"
        has_tables = f"✓ {len(page_data['tables'])} table(s)" \
                     if page_data["tables"] else "✗ tables"
        has_images = f"✓ {len(page_data['images'])} image(s)" \
                     if page_data["images"] else "✗ images"
        print(f"     Page {page_num + 1}: {has_text} | {has_tables} | {has_images}")

    fitz_doc.close()
    return "\n\n".join(all_pages)


# ── ROUTING ───────────────────────────────────────────────────────────────

def should_use_rag(pdf_path: str) -> tuple[bool, dict]:
    """Decide Direct vs RAG. Samples pages to estimate token count."""
    doc       = fitz.open(pdf_path)
    num_pages = len(doc)
    diag      = {
        "num_pages":        num_pages,
        "token_budget":     TOKEN_BUDGET,
        "decision_reason":  None,
        "estimated_tokens": None,
    }

    if num_pages > MAX_PAGES_DIRECT:
        diag["decision_reason"] = (
            f"Page count ({num_pages}) exceeds ceiling ({MAX_PAGES_DIRECT})"
        )
        doc.close()
        return True, diag

    sample_indices = set([0, num_pages - 1])
    if num_pages > 2:
        step = max(1, num_pages // 4)
        for i in range(step, num_pages - 1, step):
            sample_indices.add(i)
            if len(sample_indices) >= 5:
                break

    sample_tokens = []
    print(f"  → Sampling {len(sample_indices)} of {num_pages} page(s)...")

    for idx in sorted(sample_indices):
        try:
            page_data = extract_page(doc[idx], idx)
            content   = format_page_content(page_data)
            tokens    = count_tokens(content)
            sample_tokens.append(tokens)
            print(f"     Page {idx + 1}: ~{tokens:,} tokens")
        except Exception as e:
            import traceback
            print(f"     ⚠️  Page {idx + 1}:")
            traceback.print_exc()

    doc.close()

    if not sample_tokens:
        diag["decision_reason"] = "Sampling failed — defaulting to RAG"
        return True, diag

    avg               = sum(sample_tokens) / len(sample_tokens)
    estimated         = int(avg * num_pages)
    diag["estimated_tokens"] = estimated

    print(f"  → Avg: ~{avg:,.0f} tokens/page")
    print(f"  → Estimated total: ~{estimated:,} / {TOKEN_BUDGET:,} budget")

    if estimated > TOKEN_BUDGET:
        diag["decision_reason"] = (
            f"Estimated tokens ({estimated:,}) exceeds budget ({TOKEN_BUDGET:,})"
        )
        return True, diag

    diag["decision_reason"] = (
        f"Fits in context: ~{estimated:,} tokens, {num_pages} pages"
    )
    return False, diag


# ── SESSION INITIALISATION ────────────────────────────────────────────────

def init_session(pdf_path: str) -> Session:
    """
    Start a new session for a PDF.
    Runs extraction + routing ONCE and caches the result.
    All subsequent questions reuse the cache.
    """
    print(f"\n📋 Initialising session for: {pdf_path}")
    print("  (This runs once — all questions in this session reuse the cache)\n")

    session  = Session(pdf_path=pdf_path)
    use_rag, diag = should_use_rag(pdf_path)

    print(f"\n{'='*52}")
    print(f"  Pages:            {diag['num_pages']}")
    if diag.get("estimated_tokens"):
        print(f"  Estimated tokens: {diag['estimated_tokens']:,}")
    print(f"  Token budget:     {diag['token_budget']:,}")
    print(f"  Strategy:         {'RAG' if use_rag else 'Direct'}")
    print(f"  Reason:           {diag['decision_reason']}")
    print(f"{'='*52}\n")



    if not use_rag:
        # Extract and cache the full document content now
        print("⚡ Extracting document content into session cache...")
        content     = extract_pdf(pdf_path)
        token_count = count_tokens(content)
        print(f"  → Cached {token_count:,} tokens of document content")

        session.doc_cache = DocumentCache(
            pdf_path=pdf_path,
            content=content,
            token_count=token_count,
            use_rag=False
        )
    else:
        # RAG mode — content is in ChromaDB, no need to cache in memory
        print("🔍 RAG mode — content served from ChromaDB index")
        session.doc_cache = DocumentCache(
            pdf_path=pdf_path,
            content="",
            token_count=0,
            use_rag=True
        )

    print("\n=== CACHED CONTENT PREVIEW ===")
    print(session.doc_cache.content[:2000])
    print("=== END PREVIEW ===\n")

    return session


# ── HISTORY MANAGEMENT ────────────────────────────────────────────────────

def build_history_block(session: Session) -> str:
    """
    Build the conversation history string to prepend to each prompt.

    Structure:
      [Summarised older turns — if any]
      [Last MAX_FULL_TURNS turns in full]

    This keeps history token count bounded while preserving context.
    """
    parts = []

    if session.history_summary:
        parts.append(
            f"[SUMMARY OF EARLIER CONVERSATION]\n{session.history_summary}"
        )

    recent_turns = session.turns[-MAX_FULL_TURNS:]
    if recent_turns:
        history_lines = []
        for turn in recent_turns:
            history_lines.append(f"User: {turn.question}")
            history_lines.append(f"Assistant: {turn.answer}")
        parts.append(
            "[RECENT CONVERSATION]\n" + "\n\n".join(history_lines)
        )

    return "\n\n".join(parts)


def maybe_summarise_history(session: Session) -> None:
    """
    If conversation history is getting too long, summarise the oldest turns
    and replace them with a compact summary.

    This is exactly how production chat systems handle long conversations —
    the oldest messages are compressed while recent ones stay in full.
    """
    if len(session.turns) <= MAX_FULL_TURNS:
        return   # not long enough to need summarisation yet

    # Check if history token count exceeds limit
    history_block = build_history_block(session)
    if count_tokens(history_block) <= HISTORY_TOKEN_LIMIT:
        return   # still within budget, no need to summarise

    # Summarise all turns except the most recent MAX_FULL_TURNS
    turns_to_summarise = session.turns[:-MAX_FULL_TURNS]

    print("  → 📝 Summarising older conversation turns to save context...")

    to_summarise_text = "\n\n".join([
        f"User: {t.question}\nAssistant: {t.answer}"
        for t in turns_to_summarise
    ])

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{
            "role": "user",
            "content": f"""Summarise this conversation history concisely.
Preserve all key facts, answers, and information discussed.
This summary will be used as context for future questions.

Conversation:
{to_summarise_text}

Summary:"""
        }]
    )
    new_summary = strip_thinking(response.message.content)

    # If there was already a summary, merge it with the new one
    if session.history_summary:
        session.history_summary = (
            f"{session.history_summary}\n\n{new_summary}"
        )
    else:
        session.history_summary = new_summary

    # Keep only the recent turns in full
    session.turns = session.turns[-MAX_FULL_TURNS:]
    print(f"  → Summarised {len(turns_to_summarise)} turn(s) into history")


# ── ANSWER FUNCTIONS ──────────────────────────────────────────────────────

def answer_direct(session: Session, question: str) -> str:
    """
    Direct mode with proper chat message format.
    Document content goes in the system prompt — sent once.
    History goes as alternating user/assistant messages.
    Current question is the final user message.
    """
    # System prompt carries the document — this is the "cache"
    # In production LLM APIs this can be a true KV cache,
    # but for Ollama it's re-sent each time in the system role
    system_prompt = f"""You are a helpful assistant answering questions about a PDF document.
Answer using ONLY the document content below.
If the answer is not in the document, say so clearly.

DOCUMENT CONTENT:
{session.doc_cache.content}"""

    # Build messages array: system + alternating history + current question
    messages = [{"role": "system", "content": system_prompt}]

    # Add summarised history as a system note if it exists
    if session.history_summary:
        messages.append({
            "role":    "system",
            "content": f"Summary of earlier conversation:\n{session.history_summary}"
        })

    # Add recent turns as proper alternating user/assistant messages
    for turn in session.turns[-MAX_FULL_TURNS:]:
        messages.append({"role": "user",      "content": turn.question})
        messages.append({"role": "assistant",  "content": turn.answer})

    # Add the current question as the final user message
    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=TEXT_MODEL, messages=messages)

    content = response.message.content
    if SHOW_THINKING:
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if match:
            print("\n=== DeepSeek-R1 Reasoning ===")
            print(match.group(1).strip()[:800])

    return strip_thinking(content)


def answer_rag(session: Session, question: str) -> str:
    """
    RAG mode with proper chat message format.
    Retrieved context goes in system prompt.
    History as alternating user/assistant messages.
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
        return "⚠️  No indexed documents found. Run ingest_pdf.py first."

    if collection.count() == 0:
        return "⚠️  Vector store is empty. Run ingest_pdf.py first."

    # For follow-up questions like "Name them", we expand the query
    # with recent context so retrieval isn't fooled by pronouns
    retrieval_query = question
    if session.turns:
        last_q = session.turns[-1].question
        last_a = session.turns[-1].answer
        # Prepend last turn to give retrieval context for pronouns/references
        retrieval_query = f"{last_q} {last_a} {question}"

    results   = collection.query(
        query_texts=[retrieval_query],
        n_results=min(TOP_K * 2, collection.count()),
        include=["documents", "metadatas", "distances"]
    )
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    tokenized = [d.lower().split() for d in docs]
    bm25      = BM25Okapi(tokenized)
    bm25_raw  = bm25.get_scores(retrieval_query.lower().split())
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

    context = "\n\n---\n\n".join([
        f"[Page {r['meta'].get('page', '?')} | score={r['score']}]\n{r['doc']}"
        for r in combined
    ])

    system_prompt = f"""You are a helpful assistant answering questions about PDF documents.
Answer using ONLY the retrieved context below.
Use conversation history to resolve follow-up questions and pronouns like "them", "it", "they".
If the answer is not present, say "I cannot find this in the documents."

RETRIEVED CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]

    if session.history_summary:
        messages.append({
            "role":    "system",
            "content": f"Summary of earlier conversation:\n{session.history_summary}"
        })

    for turn in session.turns[-MAX_FULL_TURNS:]:
        messages.append({"role": "user",      "content": turn.question})
        messages.append({"role": "assistant",  "content": turn.answer})

    messages.append({"role": "user", "content": question})

    response = ollama.chat(model=TEXT_MODEL, messages=messages)
    return strip_thinking(response.message.content)


# ── MAIN SESSION LOOP ─────────────────────────────────────────────────────

def run_session(pdf_path: str) -> None:
    """
    Main interactive loop.
    - Session initialised once at the start
    - Document extracted and cached once
    - Each question reuses the cache and builds on history
    """
    session = init_session(pdf_path)

    print("\n💬 Session ready! Type 'quit' to exit.")
    print("   Type 'history' to see conversation so far.")
    print("   Type 'reset'   to start a fresh session.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not q:
            continue

        if q.lower() in ("quit", "exit", "q"):
            print(f"\nSession ended. Total questions: {session.total_questions}")
            break

        if q.lower() == "history":
            if not session.turns and not session.history_summary:
                print("  No history yet.\n")
            else:
                print("\n=== Conversation History ===")
                if session.history_summary:
                    print(f"[Summary of older turns]\n{session.history_summary}\n")
                for turn in session.turns:
                    print(f"Q{turn.turn_num}: {turn.question}")
                    print(f"A{turn.turn_num}: {turn.answer[:200]}...\n")
            continue

        if q.lower() == "reset":
            print("  Starting fresh session...\n")
            session = init_session(pdf_path)
            continue

        # ── Answer the question ──────────────────────────────────────
        session.total_questions += 1
        turn_num = session.total_questions

        print(f"\n⏳ [{turn_num}] Thinking...", end="", flush=True)

        if session.doc_cache and not session.doc_cache.use_rag:
            answer = answer_direct(session, q)
        else:
            answer = answer_rag(session, q)

        print(f"\r", end="")  # clear the thinking indicator

        # Store this turn in history
        session.turns.append(Turn(
            question=q,
            answer=answer,
            turn_num=turn_num
        ))

        # Summarise if history is getting long
        maybe_summarise_history(session)

        print(f"\nAssistant: {answer}\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf = "./pdfs/Task Handover Form - Abdul Raouf.pdf"
    if len(sys.argv) > 1:
        pdf = sys.argv[1]       # accept pdf path as argument: python query3.py my.pdf

    print("="*52)
    print("  Smart PDF Chat — Session-Aware")
    print(f"  Model:     {TEXT_MODEL}")
    print(f"  Vision:    {VISION_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}")
    print("="*52)

    run_session(pdf)