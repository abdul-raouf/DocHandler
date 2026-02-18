import gradio as gr
from pathlib import Path
import tempfile
import shutil

# Import your existing extraction logic
from exportDataToExcel import (
    extract_tables_from_pdf,
    export_to_excel,
    analyse_tables
)

def process_pdf(
    pdf_file,
    include_analysis: bool,
    progress=gr.Progress()
):
    """
    Main processing function called when user uploads a PDF.
    
    Returns:
        tuple: (excel_file_path, status_message)
    """
    if pdf_file is None:
        return None, "❌ Please upload a PDF file."
    
    try:
        # Get the uploaded file path
        pdf_path = pdf_file.name
        pdf_name = Path(pdf_path).stem
        
        progress(0.1, desc="Scanning PDF for tables...")
        
        # Extract tables (Pass 1: native, Pass 2: image-based)
        tables = extract_tables_from_pdf(pdf_path)
        
        if not tables:
            return None, "❌ No tables found in this PDF."
        
        native_count = sum(1 for t in tables if t.get("source") == "native")
        image_count  = sum(1 for t in tables if t.get("source") == "image")
        
        progress(0.7, desc=f"Exporting {len(tables)} table(s) to Excel...")
        
        # Export to Excel
        excel_path = export_to_excel(tables, pdf_path)
        
        progress(0.85, desc="Excel file created...")
        
        # Optional: Add AI analysis
        if include_analysis:
            progress(0.9, desc="Running AI analysis...")
            analyse_tables(tables, excel_path)
            analysis_msg = "\n✨ AI analysis included"
        else:
            analysis_msg = ""
        
        progress(1.0, desc="Complete!")
        
        # Success message
        status = f"""✅ **Export Complete!**
        
📊 **Tables Found:** {len(tables)} total
  - Native (text-based): {native_count}
  - Image-based (OCR): {image_count}

📁 **Excel Sheets:**
  - Contents (index)
  - {len(tables)} data sheet(s){analysis_msg}
  
👇 Download your file below
"""
        
        return str(excel_path), status
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, f"❌ **Error during processing:**\n```\n{error_detail}\n```"


# ── GRADIO INTERFACE ──────────────────────────────────────────────────────

with gr.Blocks(
    title="PDF Table Extractor",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
# 📊 PDF Table Extractor → Excel

Upload a PDF with tables and get a formatted Excel file instantly.

**Features:**
- ✅ Extracts native PDF tables (instant)
- ✅ Extracts tables from images via GLM-OCR (AI-powered)
- ✅ Professional Excel formatting with color-coding
- ✅ Optional AI analysis with insights per table
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📄 Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
            
            analysis_checkbox = gr.Checkbox(
                label="Include AI Analysis",
                value=False,
                info="Add a summary sheet with business insights per table"
            )
            
            submit_btn = gr.Button(
                "🚀 Extract Tables",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
### What happens:
1. PyMuPDF scans for native tables (instant)
2. GLM-OCR reads tables from images (30-60s per image)
3. deepseek-r1 structures and cleans data
4. Excel export with formatting
5. Optional: AI analysis added
            """)
        
        with gr.Column(scale=1):
            status_output = gr.Markdown(
                label="Status",
                value="Upload a PDF to begin..."
            )
            
            excel_output = gr.File(
                label="📥 Download Excel",
                interactive=False
            )
    
    # Examples section
    gr.Markdown("### 📝 Example PDFs")
    # gr.Examples(
    #     examples=[
    #         ["./examples/financial_report.pdf", True],
    #         ["./examples/sales_data.pdf", False],
    #     ],
    #     inputs=[pdf_input, analysis_checkbox],
    #     label="Try these sample PDFs:"
    # )
    
    # Wire up the button
    submit_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, analysis_checkbox],
        outputs=[excel_output, status_output]
    )
    
    gr.Markdown("""
---
### 🔧 Technical Stack
- **Native Tables:** PyMuPDF (`find_tables()`)
- **Image OCR:** GLM-OCR (specialized document model)
- **Structure:** deepseek-r1:8b (reasoning)
- **Analysis:** deepseek-r1:8b (business insights)
- **Export:** openpyxl + pandas

### 📖 About
Extracts tables from PDFs — both native text-based tables and tables embedded
as images (screenshots, scans). Outputs professionally formatted Excel with
optional AI-generated analysis.
    """)

# ── LAUNCH ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_port=8001)