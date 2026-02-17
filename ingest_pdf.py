import io
import fitz 
import hashlib
import base64
from pathlib import Path
import ollama
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Docling for layout-aware PDF parsing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import TableItem, PictureItem
from docling_core.types.doc.document import TextItem, SectionHeaderItem



from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── CONFIG ──────────────────────────────────────────────────────────────
VISION_MODEL  = "gemma3:4b"
TEXT_MODEL    = "deepseek-r1:8b"
EMBED_MODEL   = "mxbai-embed-large"
CHROMA_PATH   = "./chroma_db"
OLLAMA_URL    = "http://localhost:11434"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

MIN_IMAGE_WIDTH  = 150
MIN_IMAGE_HEIGHT = 150


# ── VISION HELPER ───────────────────────────────────────────────────────

def describe_image(image_bytes: bytes, doc_context: str = "") -> str:
    """
    Use gemma3:4b to describe an embedded image.
    Only called for actual embedded image objects —
    never for full page renders.
    """
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


def describe_table(table_markdown: str, doc_context: str = "") -> str:
    """
    Use deepseek-r1:8b to produce a semantic description of a table.
    Strips <think> block, stores only the final answer.
    """
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing a table from a PDF document.
Document context: {doc_context[:300]}

Table (markdown):
{table_markdown}

1. Write a 2-3 sentence plain-language summary of what this table represents.
2. List each row as a natural language sentence.
3. Note any patterns or key observations.

Write in natural language suitable for semantic search."""
        }]
    )
    content = response.message.content
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


# ── PAGE EXTRACTION (two-stage, no full-page render) ────────────────────

def extract_page(page: fitz.Page, page_num: int, doc_context: str = "") -> dict:
    """
    Two-stage extraction per page:

    Stage 1 — Free, instant, zero GPU:
        PyMuPDF reads all native text and table structure directly
        from the PDF's internal data. No AI needed.

    Stage 2 — Only for actual embedded images:
        Run vision model ONLY on specific embedded image objects.
        Never renders or processes the full page as an image.

    Returns dict with text, tables, images separately.
    """
    result = {
        "page_num": page_num + 1,
        "text":     "",
        "tables":   [],
        "images":   []
    }

    # ── STAGE 1a: Native text ─────────────────────────────────────────
    native_text = page.get_text("text").strip()
    if native_text:
        result["text"] = native_text

    # ── STAGE 1b: Native tables ───────────────────────────────────────
    try:
        tables = page.find_tables()
        for table in tables:
            df = table.to_pandas()
            if not df.empty:
                md_table = df.to_markdown(index=False)
                result["tables"].append(md_table)
    except Exception:
        pass

    # ── STAGE 2: Vision model only for embedded images ────────────────
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


# ── VECTOR STORE ─────────────────────────────────────────────────────────

def get_vectorstore():
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embeddings"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="pdf_rag",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── MAIN INGESTION ────────────────────────────────────────────────────────

def ingest_pdfs(pdf_dir: str = "./pdfs"):
    collection    = get_vectorstore()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    pdf_paths = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"Found {len(pdf_paths)} PDF(s)\n")

    for pdf_path in pdf_paths:
        print(f"📄 Processing: {pdf_path.name}")
        source    = pdf_path.name
        doc_title = pdf_path.stem.replace("_", " ").replace("-", " ").title()

        fitz_doc    = fitz.open(str(pdf_path))
        doc_context = ""

        # Build a quick doc context from first page native text
        if len(fitz_doc) > 0:
            doc_context = fitz_doc[0].get_text("text")[:1000] if len(fitz_doc) > 0 else ""

        all_ids, all_docs, all_metas = [], [], []

        for page_num in range(len(fitz_doc)):
            print(f"  → Page {page_num + 1}/{len(fitz_doc)}")
            page      = fitz_doc[page_num]
            page_data = extract_page(page, page_num, doc_context)

            has_text   = "✓ text"                            if page_data["text"]   else "✗ text"
            has_tables = f"✓ {len(page_data['tables'])} table(s)" if page_data["tables"] else "✗ tables"
            has_images = f"✓ {len(page_data['images'])} image(s)" if page_data["images"] else "✗ images"
            print(f"     {has_text} | {has_tables} | {has_images}")

            # ── Text chunks ───────────────────────────────────────────
            if page_data["text"]:
                chunks = text_splitter.split_text(page_data["text"])
                for i, chunk in enumerate(chunks):
                    contextualized = (
                        f"Document: {doc_title}\n"
                        f"Page: {page_num + 1}\n"
                        f"Chunk: {i + 1} of {len(chunks)}\n\n"
                        f"{chunk}"
                    )
                    chunk_id = hashlib.md5(
                        f"{source}-p{page_num}-text-{i}".encode()
                    ).hexdigest()
                    all_ids.append(chunk_id)
                    all_docs.append(contextualized)
                    all_metas.append({
                        "source":    source,
                        "type":      "text",
                        "page":      page_num + 1,
                        "doc_title": doc_title
                    })

            # ── Table chunks ──────────────────────────────────────────
            for i, table_md in enumerate(page_data["tables"]):
                try:
                    description = describe_table(table_md, doc_context)
                    combined    = (
                        f"Document: {doc_title}\n"
                        f"Page: {page_num + 1}\n"
                        f"TABLE {i + 1} SUMMARY:\n{description}\n\n"
                        f"RAW TABLE:\n{table_md}"
                    )
                    chunk_id = hashlib.md5(
                        f"{source}-p{page_num}-table-{i}".encode()
                    ).hexdigest()
                    all_ids.append(chunk_id)
                    all_docs.append(combined)
                    all_metas.append({
                        "source":    source,
                        "type":      "table",
                        "page":      page_num + 1,
                        "doc_title": doc_title
                    })
                except Exception as e:
                    print(f"     ⚠️  Table {i} on page {page_num + 1}: {e}")

            # ── Image chunks ──────────────────────────────────────────
            for i, img in enumerate(page_data["images"]):
                chunk_id = hashlib.md5(
                    f"{source}-p{page_num}-img-{i}".encode()
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(
                    f"Document: {doc_title}\n"
                    f"Page: {page_num + 1}\n"
                    f"IMAGE {i + 1} ({img['width']}x{img['height']}px):\n"
                    f"{img['description']}"
                )
                all_metas.append({
                    "source":    source,
                    "type":      "image",
                    "page":      page_num + 1,
                    "doc_title": doc_title
                })

        fitz_doc.close()

        if all_docs:
            print(f"\n  → Embedding {len(all_docs)} chunk(s) with {EMBED_MODEL}...")
            collection.upsert(
                ids=all_ids,
                documents=all_docs,
                metadatas=all_metas
            )
            print(f"  ✅ Done: {pdf_path.name}\n")
        else:
            print(f"  ⚠️  No content extracted from {pdf_path.name}\n")

    print(f"🎉 Ingestion complete. Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    ingest_pdfs()