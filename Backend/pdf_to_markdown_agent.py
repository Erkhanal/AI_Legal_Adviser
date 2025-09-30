from pathlib import Path
from io import BytesIO
from typing import List

from dotenv import load_dotenv
from google.genai.types import Content, Part
from pypdf import PdfReader, PdfWriter
from llm_wrapper import LLMWrapper

INPUT_DIR = Path("./input/documents")
OUTPUT_DIR = Path("./output/documents")
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
PAGES_PER_CHUNK = 5

CONVERSION_PROMPT = """SYSTEM (role = expert document-to-markdown converter):
You are a specialist in PDF structure recovery, optical character recognition, Nepali-language handling, legacy-font transliteration, and Markdown formatting for retrieval-augmented generation (RAG) pipelines.
Your mission is to produce the most faithful, clean, and semantically structured Markdown representation of the supplied PDF—even when it contains Nepali text with encoding problems.

USER INPUT
==========
<<<PDF>>>

OUTPUT TARGET
=============
Return one Markdown document that:

1. Preserves logical structure
• Map PDF headings → #, ##, …
• Convert numbered / bulleted lists faithfully.
• Render tables as GitHub-flavoured Markdown tables (| col | col |).

2. Extracts tables with high fidelity
• Detect tables even when they span page breaks; merge them into a single contiguous Markdown table, keeping the header row only once.
• Maintain column count and alignment consistently across the full table width.
• Reconstruct multi-line cells with <br> inside the cell.
• If a cell is visually merged across rows or columns, repeat the value so Markdown stays rectangular.
• Keep footnotes or captions immediately after the table as plain text.

3. Captures non-text elements
• For images/diagrams, insert ![ALT_TEXT_PLACEHOLDER](image_page-X_index-Y.png) and <!-- bbox: left, top, width, height -->.
• For mathematical formulas, output LaTeX between $ … $ or $$ … $$.

4. Eliminates noise
• Strip repeating page headers/footers, page numbers, watermarks.
• Remove hyphenated line-breaks (e.g. “informa-\ntion” → “information”).
• Collapse multiple blank lines to one blank line.

5. Annotates pages for traceability
• Insert <!----- PAGE n -------> before the first line of each page.

6. Keeps metadata
• At the top, add YAML front-matter with title, author, creation_date, page_count.

7. Handles Nepali language & encoding issues
• Detect legacy Devanagari encodings or font-based glyph mappings (e.g., Preeti, Kantipur) and convert them to proper Unicode.
• Repair common mojibake or character-swap errors using surrounding context.
• Preserve original Nepali spelling and diacritics once fixed.
• If uncertainty remains about a corrected word, wrap it in <!--? … ?-->.

8. Maintains canonical spelling
Do NOT improve grammar or paraphrase; only fix encoding.

9. Uses UTF-8 and escapes literal Markdown characters (\\*, \\_, \\|).

CONSTRAINTS
-----------
• Output nothing except the final Markdown.
• If OCR confidence for a line < 80 %, enclose that line in <!--? … ?-->.
• Preserve original page order exactly.

BEGIN.
"""

def pdf_to_byte_chunks(pdf_path: Path, pages_per_chunk: int = PAGES_PER_CHUNK) -> List[bytes]:
    reader = PdfReader(str(pdf_path))
    chunks: List[bytes] = []
    for start in range(0, len(reader.pages), pages_per_chunk):
        writer = PdfWriter()
        for p in range(start, min(start + pages_per_chunk, len(reader.pages))):
            writer.add_page(reader.pages[p])
        buf = BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    return chunks

def run_llm_on_chunk(llm: LLMWrapper, chunk_bytes: bytes) -> str:
    messages = Content(parts=[
        Part.from_text(text=CONVERSION_PROMPT),
        Part.from_bytes(data=chunk_bytes, mime_type="application/pdf")
    ])
    return llm.generate(messages).strip()

def convert_pdf(pdf_path: Path, llm: LLMWrapper) -> None:
    final_md_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
    if final_md_path.exists():
        print(f"✓ Skipping {pdf_path.name}: final Markdown already exists.")
        return

    print(f"→ Processing {pdf_path.name}")
    markdown_parts: List[str] = []
    chunks = pdf_to_byte_chunks(pdf_path)

    for idx, chunk_bytes in enumerate(chunks, start=1):
        chunk_md_path = INTERMEDIATE_DIR / f"{pdf_path.stem}_chunk{idx}.md"

        if chunk_md_path.exists():
            print(f"  • Using cached chunk {idx}/{len(chunks)}")
            md_chunk = chunk_md_path.read_text(encoding="utf-8").strip()
        else:
            print(f"  • Generating chunk {idx}/{len(chunks)}")
            md_chunk = run_llm_on_chunk(llm, chunk_bytes)
            chunk_md_path.write_text(md_chunk, encoding="utf-8")

        markdown_parts.append(md_chunk)

    combined_md = "\n\n".join(markdown_parts)
    final_md_path.write_text(combined_md, encoding="utf-8")
    print(f"  → Saved final Markdown to {final_md_path.relative_to(Path.cwd())}")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMWrapper()
    for pdf in sorted(INPUT_DIR.glob("*.pdf")):
        convert_pdf(pdf, llm)

if __name__ == "__main__":
    load_dotenv()
    main()
