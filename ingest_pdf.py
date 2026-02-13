import io
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


from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────────────
# CONFIG — swap models here if you want to experiment
# ──────────────────────────────────────────────────────
VISION_MODEL   = "gemma3:4b"          # handles images
TEXT_MODEL     = "deepseek-r1:8b"     # handles text/table reasoning
EMBED_MODEL    = "nomic-embed-text"   # handles embeddings
CHROMA_PATH    = "./chroma_db"
OLLAMA_URL     = "http://localhost:11434"


# ──────────────────────────────────────────────────────
# STEP 1: Describe images using gemma3:4b (vision model)
# ──────────────────────────────────────────────────────
def describe_image(image_bytes: bytes, doc_context: str = "") -> str:
    """
    gemma3:4b accepts raw image bytes directly via the Ollama Python SDK.
    No base64 encoding needed — the SDK handles it.
    """
    prompt = f"""You are analyzing an image extracted from a PDF document.
Document context: {doc_context[:400]}

Describe this image in detail. If it contains:
- Charts/graphs: describe the trend, axes, and key values
- Diagrams: explain the structure and relationships  
- Tables embedded as images: extract the data as text
- Photos/illustrations: describe what is depicted

Be thorough — your description will be used for semantic search."""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_bytes]   # pass raw bytes directly
        }]
    )
    return response.message.content


# ──────────────────────────────────────────────────────
# STEP 2: Describe tables using deepseek-r1:8b
# deepseek-r1 uses chain-of-thought reasoning, which is
# excellent for understanding table structure and meaning.
# We strip the <think> block from output for clean storage.
# ──────────────────────────────────────────────────────
def describe_table(table_markdown: str, doc_context: str = "") -> str:
    """
    deepseek-r1 produces <think>...</think> reasoning blocks.
    We extract only the final answer after the thinking block.
    """
    prompt = f"""You are analyzing a table from a PDF document.
Document context: {doc_context[:400]}

Table (markdown format):
{table_markdown}

1. Write a 2-3 sentence plain-language summary of what this table represents.
2. List the most important data points as natural sentences.

Write in natural language suitable for semantic search."""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.message.content

    # Strip deepseek-r1's <think>...</think> reasoning block
    # We only want the final answer for clean storage in the vector DB
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()

    return content


# ──────────────────────────────────────────────────────
# STEP 3: Set up ChromaDB with Ollama embeddings
# nomic-embed-text runs locally — no OpenAI calls at all
# ──────────────────────────────────────────────────────
def get_vectorstore():
    embedding_fn = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url=f"{OLLAMA_URL}/api/embeddings"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="pdf_rag",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )
    return collection


# ──────────────────────────────────────────────────────
# STEP 4: Configure Docling for layout-aware PDF parsing
# ──────────────────────────────────────────────────────
def get_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    format_option = PdfFormatOption(
        pipeline_options=pipeline_options,
        backend=PyPdfiumDocumentBackend  # correct backend
    )

    return DocumentConverter(
        format_options={InputFormat.PDF: format_option}
    )

# ──────────────────────────────────────────────────────
# STEP 5: Main ingestion pipeline
# ──────────────────────────────────────────────────────
def ingest_pdfs(pdf_dir: str = "./pdfs"):
    converter = get_converter()
    collection = get_vectorstore()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    pdf_paths = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"Found {len(pdf_paths)} PDF(s)\n")

    for pdf_path in pdf_paths:
        print(f"📄 Processing: {pdf_path.name}")
        result = converter.convert(str(pdf_path))
        doc = result.document
        doc_context = doc.export_to_markdown()[:2000]
        source = pdf_path.name

        all_ids, all_docs, all_metas = [], [], []

        # ── A) TEXT chunks ──────────────────────────────
        print("  → Chunking text...")
        text_content = doc.export_to_markdown()
        chunks = text_splitter.split_text(text_content)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source}-text-{i}".encode()).hexdigest()
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metas.append({"source": source, "type": "text", "chunk_index": i})
        print(f"     → {len(chunks)} text chunks")

        # ── B) TABLES (described by deepseek-r1) ────────
        print("  → Processing tables with deepseek-r1:8b...")
        table_count = 0
        for element in doc.iterate_items():
            item, _ = element
            if hasattr(item, 'label') and item.label == DocItemLabel.TABLE:
                try:
                    raw_table = item.export_to_markdown()
                    description = describe_table(raw_table, doc_context)
                    combined = f"TABLE SUMMARY:\n{description}\n\nRAW TABLE:\n{raw_table}"
                    chunk_id = hashlib.md5(
                        f"{source}-table-{table_count}".encode()
                    ).hexdigest()
                    all_ids.append(chunk_id)
                    all_docs.append(combined)
                    all_metas.append({
                        "source": source,
                        "type": "table",
                        "raw": raw_table
                    })
                    table_count += 1
                except Exception as e:
                    print(f"     ⚠️ Skipped table: {e}")
        print(f"     → {table_count} tables")

        # ── C) IMAGES (described by gemma3:4b) ──────────
        print("  → Processing images with gemma3:4b...")
        image_count = 0
        for element in doc.iterate_items():
            item, _ = element
            if hasattr(item, 'label') and item.label == DocItemLabel.PICTURE:
                try:
                    pil_image = item.image.pil_image
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    image_bytes = buf.getvalue()

                    description = describe_image(image_bytes, doc_context)
                    chunk_id = hashlib.md5(
                        f"{source}-image-{image_count}".encode()
                    ).hexdigest()
                    all_ids.append(chunk_id)
                    all_docs.append(f"IMAGE DESCRIPTION:\n{description}")
                    all_metas.append({"source": source, "type": "image"})
                    image_count += 1
                except Exception as e:
                    print(f"     ⚠️ Skipped image: {e}")
        print(f"     → {image_count} images")

        # ── D) Add all chunks to ChromaDB in one batch ──
        print(f"  → Embedding {len(all_docs)} chunks with nomic-embed-text...")
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas
        )
        print(f"  ✅ Done: {pdf_path.name}\n")

    total = collection.count()
    print(f"🎉 Ingestion complete. Total chunks in DB: {total}")


if __name__ == "__main__":
    ingest_pdfs()