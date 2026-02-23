#!/usr/bin/env python3
"""
Document ingestion pipeline for Connectome Lab RAG.
Loads PDFs, chunks them with academic-paper-aware splitting, and builds a FAISS index.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers" / "pdf"
INDEX_DIR = PROJECT_ROOT / "rag" / "data" / "faiss_index"
METADATA_PATH = PROJECT_ROOT / "rag" / "data" / "doc_metadata.json"


def load_pdfs(papers_dir: Path) -> list:
    """Load all PDFs using LangChain's PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader

    all_docs = []
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {papers_dir}")

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["paper_title"] = pdf_path.stem.replace("_", " ")
            all_docs.extend(docs)
            logger.info(f"  Loaded {len(docs)} pages from {pdf_path.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {pdf_path.name}: {e}")

    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    """Split documents into chunks optimized for academic papers."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",       # paragraph
            "\nAbstract",  # section headers
            "\nIntroduction",
            "\nMethods",
            "\nResults",
            "\nDiscussion",
            "\nConclusion",
            "\nReferences",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    logger.info(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def build_faiss_index(
    chunks: list,
    model_name: str = "nvidia/nv-embedqa-e5-v5",
    nvidia_api_key: Optional[str] = None,
    use_local: bool = True,
) -> None:
    """Build FAISS index from document chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if use_local:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info("Using local embedding model: intfloat/e5-large-v2")
    else:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        embedder = NVIDIAEmbeddings(
            model=model_name,
            api_key=nvidia_api_key,
        )
        logger.info(f"Using NVIDIA API embedding: {model_name}")

    from langchain_community.vectorstores import FAISS

    logger.info("Building FAISS index (this may take a while)...")
    batch_size = 100
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedder)
        else:
            batch_store = FAISS.from_documents(batch, embedder)
            vectorstore.merge_from(batch_store)
        logger.info(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"FAISS index saved to {INDEX_DIR}")

    metadata = {
        "total_chunks": len(chunks),
        "embedding_model": "intfloat/e5-large-v2" if use_local else model_name,
        "chunk_size": 512,
        "chunk_overlap": 64,
        "papers_dir": str(PAPERS_DIR),
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Metadata saved to {METADATA_PATH}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vector store")
    parser.add_argument("--nvidia-api-key", type=str, default=None, help="NVIDIA API key for cloud embeddings")
    parser.add_argument("--use-local", action="store_true", default=True, help="Use local embedding model")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()

    documents = load_pdfs(PAPERS_DIR)
    if not documents:
        logger.error("No documents loaded. Run download_papers.py first.")
        sys.exit(1)

    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    build_faiss_index(chunks, nvidia_api_key=args.nvidia_api_key, use_local=args.use_local)
    logger.info("Ingestion pipeline complete!")


if __name__ == "__main__":
    main()
