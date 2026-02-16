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

# ──────────────────────────────────────────────────────
# CONFIG — swap models here if you want to experiment
# ──────────────────────────────────────────────────────
VISION_MODEL   = "gemma3:4b"          # handles images
TEXT_MODEL     = "deepseek-r1:8b"     # handles text/table reasoning
EMBED_MODEL    = "mxbai-embed-large"   # handles embeddings
CHROMA_PATH    = "./chroma_db"
OLLAMA_URL     = "http://localhost:11434"

CHUNK_SIZE    = 400    # smaller = more precise retrieval
CHUNK_OVERLAP = 80     # ~20% overlap to preserve context at boundaries


# Text fragments from Docling that are captions/noise, not real content
JUNK_PHRASES = [
    "here is the requested image",
    "this is the table",
    "<!-- image -->",
    "figure ",
    "table ",
]

# Minimum character length for a text item to be worth keeping
MIN_TEXT_LENGTH = 20

# Minimum image dimensions to skip icons/bullets/decorative elements
MIN_IMAGE_WIDTH  = 100
MIN_IMAGE_HEIGHT = 100


# ──────────────────────────────────────────────────────
# STEP 1: Describe images using gemma3:4b (vision model)
# ──────────────────────────────────────────────────────
def describe_image(image_bytes: bytes, doc_context: str = "") -> str:
    """
    Use gemma3:4b (vision) to convert an image into a rich text description.
    Prompt is designed to handle UI screenshots, charts, diagrams, and photos.
    """
    prompt = f"""You are analyzing an image extracted from a PDF document.
Document context: {doc_context[:400]}

Analyze this image carefully. It could be any of the following:
- A UI screenshot or application window  → transcribe ALL visible text exactly, describe every section, button, label
- A chart or graph                       → describe axes, values, trends, title
- A diagram                             → describe structure, labels, relationships
- A photo or illustration               → describe what is shown in detail
- A table rendered as an image          → extract ALL rows and columns as text

Critical instructions:
1. Transcribe ANY text visible in the image EXACTLY as it appears.
2. Preserve all headings, labels, section names, and numerical values.
3. Be exhaustive — your description is the ONLY way this image's content can be searched.
4. Start directly with the content — do not say 'I can see an image of...'"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_bytes]
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
    Use deepseek-r1:8b to produce a semantic, searchable description of a table.
    Strip the <think> block — we only store the final answer.
    """
    prompt = f"""You are analyzing a table extracted from a PDF document.
Document context: {doc_context[:400]}

Table (markdown):
{table_markdown}

Tasks:
1. Write a 2-3 sentence plain-language summary of what this table represents.
2. List each row as a natural language sentence (e.g. "Abc holds the role of Software Engineer.").
3. Note any patterns, totals, or key observations.

Write in natural language — this will be used for semantic search."""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.message.content
    # Strip deepseek-r1's chain-of-thought block — keep only final answer
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content



# ──────────────────────────────────────────────────────
# STEP 3: Set up ChromaDB with Ollama embeddings
# nomic-embed-text runs locally — no OpenAI calls at all
# ──────────────────────────────────────────────────────
def get_vectorstore():
    """
    Returns a ChromaDB collection using mxbai-embed-large via Ollama.
    Uses cosine similarity — best for text retrieval tasks.
    """
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
    converter    = get_converter()
    collection   = get_vectorstore()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]  # respect sentence boundaries
    )

    pdf_paths = list(Path(pdf_dir).glob("**/*.pdf"))
    print(f"Found {len(pdf_paths)} PDF(s)\n")

    for pdf_path in pdf_paths:
        print(f"📄 Processing: {pdf_path.name}")
        result      = converter.convert(str(pdf_path))
        doc         = result.document
        doc_context = doc.export_to_markdown()[:2000]
        source      = pdf_path.name
        doc_title   = pdf_path.stem.replace("_", " ").replace("-", " ").title()

        all_ids, all_docs, all_metas = [], [], []

        # ── A) TEXT ──────────────────────────────────────────────────────
        # FIX: Extract ONLY pure text/heading items — skip table & image placeholders.
        # Previously we used doc.export_to_markdown() which mixed everything together,
        # causing <!-- image --> noise and raw table markdown to pollute text chunks.
        print("  → Extracting clean text...")
        text_only_parts = []

        for item, _ in doc.iterate_items():
            if isinstance(item, (TextItem, SectionHeaderItem)):
                text = item.text.strip() if item.text else ""
                if len(text) < MIN_TEXT_LENGTH:
                    continue
                if any(junk in text.lower() for junk in JUNK_PHRASES):
                    continue
                text_only_parts.append(text)

        if text_only_parts:
            text_content = "\n\n".join(text_only_parts)
            chunks = text_splitter.split_text(text_content)
            for i, chunk in enumerate(chunks):
                contextualized = (
                    f"Document: {doc_title}\n"
                    f"Chunk {i + 1} of {len(chunks)}\n\n"
                    f"{chunk}"
                )
                chunk_id = hashlib.md5(
                    f"{source}-text-{i}".encode()
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(contextualized)
                all_metas.append({
                    "source":      source,
                    "type":        "text",
                    "chunk_index": i,
                    "doc_title":   doc_title
                })
            print(f"     → {len(chunks)} text chunk(s)")
        else:
            print("     → No clean text found (image-only or caption-only PDF)")


        # ── B) TABLES ────────────────────────────────────────────────────
        # Use doc.tables (Docling's direct property) — avoids label-checking.
        # Store BOTH the LLM semantic description AND the raw markdown table
        # so the LLM at query time can access exact values if needed.
        print(f"  → Processing {len(doc.tables)} table(s) with {TEXT_MODEL}...")
        for i, table in enumerate(doc.tables):
            try:
                # Pass doc object — fixes deprecation warning
                raw_table = table.export_to_markdown(doc)

                description = describe_table(raw_table, doc_context)

                # Store description + raw table together for best of both worlds:
                # - Description helps with semantic/conceptual queries
                # - Raw table helps with exact value lookups
                combined = (
                    f"Document: {doc_title}\n"
                    f"TABLE {i + 1} SEMANTIC SUMMARY:\n{description}\n\n"
                    f"RAW TABLE DATA:\n{raw_table}"
                )
                chunk_id = hashlib.md5(f"{source}-table-{i}".encode()).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(combined)
                all_metas.append({
                    "source":    source,
                    "type":      "table",
                    "table_index": i,
                    "doc_title": doc_title,
                    "raw":       raw_table      # available for exact retrieval
                })
                print(f"     → Table {i} processed")
            except Exception as e:
                print(f"     ⚠️  Skipped table {i}: {e}")

        # ── C) IMAGES ────────────────────────────────────────────────────
        # Use doc.pictures (Docling's direct property).
        # IMPROVEMENT: Enhanced prompt captures UI screenshots, not just photos.
        # The description becomes the searchable text for this image.
        print(f"  → Processing {len(doc.pictures)} image(s) with {VISION_MODEL}...")
        seen_hashes  = set()
        image_count  = 0

        for i, picture in enumerate(doc.pictures):
            try:
                pil_image = picture.get_image(doc)
                if pil_image is None:
                    continue

                w, h = pil_image.size
                if w < 800 or h < 800:
                    scale  = max(800 / w, 800 / h)
                    new_w  = int(w * scale)
                    new_h  = int(h * scale)
                    from PIL import Image as PILImage
                    pil_image = pil_image.resize(
                        (new_w, new_h),
                        PILImage.Resampling.LANCZOS   # high-quality upscale
                    )

                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                img_hash = hashlib.md5(image_bytes).hexdigest()
                seen_hashes.add(img_hash)

                description = describe_image(image_bytes, doc_context)
                chunk_id = hashlib.md5(
                    f"{source}-image-{i}".encode()
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(
                    f"Document: {doc_title}\n"
                    f"IMAGE {i + 1} DESCRIPTION:\n{description}"
                )
                all_metas.append({
                    "source":      source,
                    "type":        "image",
                    "image_index": i,
                    "doc_title":   doc_title,
                    "extractor":   "docling"
                })
                image_count += 1
            except Exception as e:
                print(f"     ⚠️  Skipped image {i}: {e}")

        print(f"     → {image_count} image(s) via Docling")

       # ── D) PAGE RENDER FALLBACK via PyMuPDF ───────────────────────────────
        # Instead of extracting embedded image objects (which can miss screenshots),
        # render each full PDF page as a high-res image and describe it.
        # This is the most robust approach for PDFs with UI screenshots, diagrams,
        # or any content that isn't a clean embedded image object.
        print("  → Rendering full pages via PyMuPDF for complete visual coverage...")
        fitz_doc       = fitz.open(str(pdf_path))
        fallback_count = 0

        for page_num in range(len(fitz_doc)):
            try:
                page = fitz_doc[page_num]

                # Render at 2x scale — balances quality vs. memory
                # Increase to 3.0 for very dense pages (more detail for vision model)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert pixmap → PNG bytes
                image_bytes = pix.tobytes("png")

                # Skip near-blank pages (mostly whitespace)
                # A blank page compressed PNG is typically < 5KB
                if len(image_bytes) < 5000:
                    print(f"     → Page {page_num + 1} appears blank, skipping")
                    continue

                description = describe_image(image_bytes, doc_context)

                chunk_id = hashlib.md5(
                    f"{source}-pagerender-p{page_num}".encode()
                ).hexdigest()
                all_ids.append(chunk_id)
                all_docs.append(
                    f"Document: {doc_title}\n"
                    f"FULL PAGE {page_num + 1} VISUAL DESCRIPTION:\n{description}"
                )
                all_metas.append({
                    "source":    source,
                    "type":      "image",
                    "page":      page_num + 1,
                    "doc_title": doc_title,
                    "extractor": "pymupdf_pagerender"
                })
                fallback_count += 1
                print(f"     → Page {page_num + 1} rendered and described")

            except Exception as e:
                print(f"     ⚠️  Failed to render page {page_num + 1}: {e}")

        fitz_doc.close()
        print(f"     → {fallback_count} page render(s) added")

        # ── E) UPSERT to ChromaDB ─────────────────────────────────────
        if all_docs:
            print(
                f"  → Embedding {len(all_docs)} total chunks "
                f"with {EMBED_MODEL}..."
            )
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