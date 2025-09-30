import hashlib
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from google.genai.types import Content, Part

from llm_wrapper import LLMWrapper
from vector_store_wrapper import VectorStoreWrapper
from embedding_engine_wrapper import EmbeddingEngineWrapper

INPUT_DIR = Path("./output/documents")
PROCESSED_DIR = Path("./processed/documents")
PAGE_DELIM_REGEX = re.compile(r"<\!-----\s*PAGE\s+\d+\s*------->", re.IGNORECASE)

# Chunking parameters
CHUNK_PAGE_COUNT = 5  # pages per chunk
CHUNK_OVERLAP = 1     # overlap in pages between consecutive chunks
_STEP = CHUNK_PAGE_COUNT - CHUNK_OVERLAP

# Embedding parameters
EMBED_BATCH_SIZE = 32  # number of texts per embedder batch call

logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")


def chunk_text(text: str) -> List[str]:
    """Split markdown by page delimiter, then group pages into overlapping chunks."""
    pages = [p.strip() for p in re.split(PAGE_DELIM_REGEX, text) if p.strip()]
    if len(pages) <= CHUNK_PAGE_COUNT:
        return ["\n".join(pages)] if pages else []

    chunks: List[str] = []
    for start in range(0, len(pages), _STEP):
        slice_ = pages[start : start + CHUNK_PAGE_COUNT]
        if not slice_:
            break
        chunks.append("\n".join(slice_))
        if start + CHUNK_PAGE_COUNT >= len(pages):
            break
    return chunks


def deterministic_id(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()


def build_prompt(chunk: str, file_name: str, include_identity: bool) -> str:
    identity_section = (
        "First, inspect the chunk for an official document title, short title, or citation (e.g. \"Constitution of Nepal 2015\", \"Contract Act 2056 (1999)\"). "
        "If found, start your answer with 'DocumentTitle: <extracted title>'.\n\n"
        if include_identity
        else ""
    )

    return (
        f"""You are a legal-domain summarization assistant.

{identity_section}Then, in concise bullet points, extract:
• section / article numbers
• definitions
• obligations
• rights
• penalties
• jurisdiction
• effective / promulgation dates
• parties or entities involved
• keywords crucial for search.

**Important:** Write the entire summary **only in English**, even if the source text is Nepali. When translating Nepali legal terms, provide an English equivalent followed by the original transliteration in parentheses if the original wording carries legal significance.

The goal is to build a vector store that can answer precise user queries.

Always append a final line 'SourceDoc: {file_name}'.

{chunk}
"""
    )


def batch_embed(embedder: EmbeddingEngineWrapper, texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        emb_objs = embedder.embed(batch)
        if emb_objs is None or len(emb_objs) != len(batch):
            logging.error("Embedding batch failed (size %d)", len(batch))
            raise RuntimeError("Embedding failure")
        vectors.extend([e.values for e in emb_objs])
    return vectors


def main():
    if not INPUT_DIR.exists():
        logging.error("Input directory %s does not exist", INPUT_DIR)
        return
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = VectorStoreWrapper()
    embedder = EmbeddingEngineWrapper()
    llm_wrapper = LLMWrapper(model_name="gemini-2.5-flash")

    for md_path in INPUT_DIR.glob("*.md"):
        processed_target = PROCESSED_DIR / md_path.name
        if processed_target.exists():
            logging.info("Skipping %s; already processed", md_path.name)
            continue

        logging.info("Processing %s", md_path.name)
        text = md_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        # chunks = chunks[:3]

        summaries: List[str] = []
        embed_inputs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        # 1) Summarize each chunk sequentially
        for idx, chunk in enumerate(chunks):
            logging.info(f"Loading chunk {idx}/{len(chunks)}")
            prompt = build_prompt(chunk, md_path.name, include_identity=(idx == 0))
            summary = llm_wrapper.generate(prompt)
            summaries.append(summary)
            embed_inputs.append(f"Summary:\n{summary}\n\nDocument Chunk:\n{chunk}")
            metadatas.append({"document": chunk, "summary": summary})
            ids.append(deterministic_id(f"{md_path.name}-chunk{idx}"))

        if not summaries:
            logging.warning("No summaries produced for %s", md_path.name)
            continue

        # 2) Batch‑embed all (summary + chunk) pairs
        try:
            embeddings = batch_embed(embedder, embed_inputs)
        except RuntimeError:
            logging.error("Skipping %s due to embedding errors", md_path.name)
            continue

        # 3) Delete old vectors and upsert new ones
        vector_store.delete(ids=ids)
        logging.info("Deleted any existing embeddings for %s", md_path.name)

        vector_store.upsert(embeddings=embeddings, metadatas=metadatas, ids=ids)
        logging.info("Upserted %d chunks for %s into vector store", len(embeddings), md_path.name)

        # 4) Move processed file
        shutil.copy2(md_path, processed_target)
        logging.info("Moved %s to processed directory", md_path.name)


if __name__ == "__main__":
    load_dotenv()
    main()
