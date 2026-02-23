#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

print_header() {
    echo "=============================================="
    echo "  SNU Connectome Lab — RAG Pipeline Runner"
    echo "=============================================="
    echo ""
}

check_deps() {
    echo "[1/4] Checking dependencies..."
    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 not found"; exit 1
    fi
    
    local missing=()
    python3 -c "import langchain" 2>/dev/null || missing+=("langchain")
    python3 -c "import faiss" 2>/dev/null || missing+=("faiss-cpu")
    python3 -c "import gradio" 2>/dev/null || missing+=("gradio")
    python3 -c "import aiohttp" 2>/dev/null || missing+=("aiohttp")
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo "  Installing missing packages..."
        pip install -r requirements.txt
    else
        echo "  All dependencies OK"
    fi
}

download_papers() {
    echo ""
    echo "[2/4] Downloading papers..."
    local pdf_count
    pdf_count=$(find papers/pdf -name "*.pdf" 2>/dev/null | wc -l)
    
    if [ "$pdf_count" -gt 10 ]; then
        echo "  Found $pdf_count PDFs already downloaded. Skipping."
    else
        python3 scripts/download_papers.py
    fi
}

build_index() {
    echo ""
    echo "[3/4] Building FAISS index..."
    if [ -d "rag/data/faiss_index" ] && [ -f "rag/data/faiss_index/index.faiss" ]; then
        echo "  FAISS index already exists. Skipping."
        echo "  (Delete rag/data/faiss_index/ to rebuild)"
    else
        python3 rag/src/ingest.py --use-local
    fi
}

launch_chatbot() {
    echo ""
    echo "[4/4] Launching chatbot..."
    echo "  Mode: ${RAG_MODE:-local}"
    echo "  URL:  http://localhost:7860"
    echo ""
    
    local args="--host 0.0.0.0 --port 7860"
    if [ -n "${RAG_MODE:-}" ]; then
        args="$args --mode $RAG_MODE"
    fi
    if [ "${RAG_SHARE:-}" = "1" ]; then
        args="$args --share"
    fi
    
    python3 rag/src/chatbot.py $args
}

print_header
check_deps
download_papers
build_index
launch_chatbot
