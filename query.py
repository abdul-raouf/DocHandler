import re
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

EMBED_MODEL  = "nomic-embed-text"
TEXT_MODEL   = "deepseek-r1:8b"
CHROMA_PATH  = "./chroma_db"
OLLAMA_URL   = "http://localhost:11434"


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
    """
    deepseek-r1 wraps its chain-of-thought in <think>...</think>.
    This is useful for debugging but we strip it for clean output.
    Set SHOW_THINKING = True below to see the full reasoning chain.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


SHOW_THINKING = False   # Set True to see deepseek-r1's reasoning chain


def query_rag(question: str, top_k: int = 5):
    collection = get_collection()

    # Retrieve top-k semantically similar chunks
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Build labelled context from all content types
    context_parts = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        label = meta.get("type", "text").upper()
        source = meta.get("source", "unknown")
        similarity = round(1 - dist, 3)   # cosine distance → similarity
        context_parts.append(
            f"[{label} | {source} | similarity: {similarity}]\n{doc}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Prompt — deepseek-r1 responds well to explicit step-by-step instructions
    prompt = f"""You are a helpful assistant answering questions from PDF documents.
Answer the question using ONLY the context below.
The context may include text, table summaries, and image descriptions.

If the answer involves a table: present the data clearly.
If the answer involves an image: describe what the image shows.
If the answer is not in the context: say "I cannot find this in the documents."

Context:
{context}

Question: {question}

Answer:"""

    print("\n⏳ Thinking with deepseek-r1:8b...\n")

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_answer = response.message.content

    # Display retrieved sources
    print("=== Retrieved Sources ===")
    for i, (doc, meta, dist) in enumerate(
        zip(docs, metas, distances), 1
    ):
        t = meta.get("type", "text").upper()
        s = meta.get("source", "?")
        sim = round(1 - dist, 3)
        print(f"\n[{i}] {t} | {s} | similarity={sim}")
        print(doc[:250] + ("..." if len(doc) > 250 else ""))

    # Optionally show deepseek-r1's chain of thought
    if SHOW_THINKING and "<think>" in raw_answer:
        think_block = re.search(
            r"<think>(.*?)</think>", raw_answer, re.DOTALL
        )
        if think_block:
            print("\n=== DeepSeek-R1 Reasoning Chain ===")
            print(think_block.group(1).strip()[:1000] + "...")

    print("\n=== Answer ===")
    clean_answer = strip_thinking(raw_answer)
    print(clean_answer)
    return clean_answer


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        if q:
            query_rag(q)