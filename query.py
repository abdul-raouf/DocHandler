import re
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from rank_bm25 import BM25Okapi

# ── CONFIG ─────────────────────────────────────────────────────────────
TEXT_MODEL  = "deepseek-r1:8b"
EMBED_MODEL = "mxbai-embed-large"
CHROMA_PATH = "./chroma_db"
OLLAMA_URL  = "http://localhost:11434"

TOP_K        = 6      # fetch more candidates before reranking
FINAL_TOP_K  = 3      # pass this many chunks to the LLM after reranking
SHOW_THINKING = False  # set True to see deepseek-r1's reasoning chain


# ── HELPERS ────────────────────────────────────────────────────────────

def get_collection():
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embeddings"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(
        name="pdf_rag",
        embedding_function=embedding_fn
    )


def strip_thinking(text: str) -> str:
    """Remove deepseek-r1's <think>...</think> block from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def hybrid_search(question: str, collection, top_k: int = TOP_K) -> list[dict]:
    """
    Combine vector similarity search (ChromaDB) with BM25 keyword search.

    Why hybrid?
    - Vector search is great for semantic/conceptual queries
      e.g. "who manages the team?" → finds "Manager" even if word not in query
    - BM25 is great for exact keyword matches
      e.g. "who is Ghi?" → scores 'Ghi' highly even if semantically distant

    We fetch top_k*2 candidates from vector search, score them all with BM25,
    then combine: final_score = 0.6 * vector_score + 0.4 * bm25_score
    """
    # Step 1: Vector search — fetch 2x candidates for reranking pool
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k * 2, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return []

    # Step 2: BM25 keyword scoring over the same candidate pool
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25             = BM25Okapi(tokenized_corpus)
    bm25_scores      = bm25.get_scores(question.lower().split())

    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = [s / bm25_max for s in bm25_scores]

    # Step 3: Combine scores
    # Vector similarity = 1 - cosine_distance (already in [0,1])
    combined = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        vector_score = 1 - dist                          # cosine similarity
        bm25_score   = bm25_norm[i]
        final_score  = 0.6 * vector_score + 0.4 * bm25_score   # tunable weights

        combined.append({
            "document":     doc,
            "metadata":     meta,
            "vector_score": round(vector_score, 3),
            "bm25_score":   round(bm25_score, 3),
            "final_score":  round(final_score, 3)
        })

    # Step 4: Sort by combined score descending, return top_k
    combined.sort(key=lambda x: x["final_score"], reverse=True)
    return combined[:top_k]


def build_context(ranked_results: list[dict]) -> str:
    """Format retrieved chunks into a labelled context block for the LLM."""
    parts = []
    for i, r in enumerate(ranked_results, 1):
        content_type = r["metadata"].get("type", "text").upper()
        source       = r["metadata"].get("source", "unknown")
        score        = r["final_score"]
        parts.append(
            f"[{i}] {content_type} | {source} | relevance={score}\n"
            f"{r['document']}"
        )
    return "\n\n---\n\n".join(parts)


# ── MAIN QUERY ─────────────────────────────────────────────────────────

def query_rag(question: str):
    collection = get_collection()

    if collection.count() == 0:
        print("⚠️  Vector store is empty. Run ingest_pdf.py first.")
        return

    # Step 1: Hybrid search (vector + BM25)
    ranked = hybrid_search(question, collection, top_k=TOP_K)

    if not ranked:
        print("No relevant documents found.")
        return

    # Step 2: Take top FINAL_TOP_K after hybrid ranking
    top_results = ranked[:FINAL_TOP_K]
    context     = build_context(top_results)

    print(f"Found {len(top_results)} relevant document chunks. Context->", context[:500], "...\n")

    # Step 3: Build prompt for deepseek-r1
    # Explicit instructions handle all content types (text, table, image)
    prompt = f"""You are a precise assistant answering questions from PDF documents.
Use ONLY the context below. Do not invent information not present in the context.

Guidelines:
- If the answer involves TABLE data: present it clearly, preserve exact values.
- If the answer involves an IMAGE: describe what was found in the image description.
- If the answer is not in the context: say exactly "I cannot find this in the provided documents."
- Be concise and direct.

Context:
{context}

Question: {question}

Answer:"""

    print(f"\n⏳ Reasoning with {TEXT_MODEL}...\n")

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_answer = response.message.content

    # ── Display retrieved sources ──
    print("=== Retrieved Sources ===")
    for i, r in enumerate(top_results, 1):
        t     = r["metadata"].get("type", "text").upper()
        s     = r["metadata"].get("source", "?")
        vs    = r["vector_score"]
        bs    = r["bm25_score"]
        fs    = r["final_score"]
        print(f"\n[{i}] {t} | {s}")
        print(f"    vector={vs}  bm25={bs}  combined={fs}")
        print(f"    {r['document'][:200]}...")

    # ── Optionally show deepseek-r1's reasoning chain ──
    if SHOW_THINKING:
        match = re.search(r"<think>(.*?)</think>", raw_answer, re.DOTALL)
        if match:
            print("\n=== DeepSeek-R1 Reasoning Chain (SHOW_THINKING=True) ===")
            print(match.group(1).strip()[:1000])

    # ── Final answer ──
    print("\n=== Answer ===")
    clean = strip_thinking(raw_answer)
    print(clean)
    return clean


# ── ENTRY POINT ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PDF RAG — Local (deepseek-r1 + gemma3 + mxbai-embed-large)")
    print("Type 'quit' to exit | Set SHOW_THINKING=True to see reasoning\n")
    while True:
        q = input("Ask a question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            query_rag(q)