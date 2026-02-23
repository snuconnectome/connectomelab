# SNU Connectome Lab — RAG Research Chatbot

Retrieval-Augmented Generation chatbot for Prof. Jiook Cha's research publications at Seoul National University Connectome Lab. External users can query the lab's 82+ publications through a conversational interface powered by NVIDIA Nemotron.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐
│  Gradio UI  │───▶│   Retriever  │───▶│  FAISS VectorDB   │
│  (port 7860)│    │  (top-k=5)   │    │  (E5-large embed) │
└──────┬──────┘    └──────────────┘    └───────────────────┘
       │                                        ▲
       ▼                                        │
┌──────────────┐                      ┌─────────┴─────────┐
│  LLM Engine  │                      │   PDF Ingestion    │
│  (Nemotron / │                      │   (PyPDF + chunk)  │
│   Phi-3.5)   │                      └───────────────────┘
└──────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download papers → build index → launch chatbot
bash scripts/run.sh
```

## Inference Modes

| Mode | GPU Required | Config `mode` | Description |
|------|-------------|---------------|-------------|
| **Local** | Yes (12GB+) | `local` | Phi-3.5-mini on local GPU |
| **NVIDIA Cloud** | No | `nvidia_cloud` | Nemotron via build.nvidia.com API |
| **NVIDIA NIM** | Yes (24GB+) | `nvidia_nim` | Self-hosted NIM container |

### NVIDIA Cloud (no GPU needed)

```bash
export NVIDIA_API_KEY=nvapi-...  # from build.nvidia.com
RAG_MODE=nvidia_cloud bash scripts/run.sh
```

### NVIDIA NIM (self-hosted)

```bash
export NGC_API_KEY=...
docker compose --profile nim --profile app up -d
```

## Project Structure

```
connectomelab/
├── papers/
│   ├── pdf/              # Downloaded research papers
│   └── metadata/         # Download report
├── rag/
│   ├── config/           # settings.json
│   ├── data/             # FAISS index (auto-generated)
│   └── src/
│       ├── ingest.py     # PDF → chunks → FAISS
│       └── chatbot.py    # Gradio RAG chatbot
├── scripts/
│   ├── download_papers.py    # Async parallel PDF downloader
│   ├── run.sh                # One-command pipeline runner
│   └── setup_workbench.sh    # NVIDIA AI Workbench setup
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Configuration

Edit `rag/config/settings.json`:

```json
{
  "mode": "nvidia_cloud",
  "nvidia_api_key": "nvapi-...",
  "nvidia_model": "nvidia/nemotron-3-nano-30b-a3b",
  "temperature": 0.2,
  "top_k": 5
}
```

## Recommended Models (2026)

| Component | Model | Notes |
|-----------|-------|-------|
| LLM (low latency) | `nvidia/llama-3.1-nemotron-nano-8b-v1` | Single GPU, fast |
| LLM (accuracy) | `nvidia/nemotron-3-nano-30b-a3b` | MoE, 3.5B active params |
| Embedding | `intfloat/e5-large-v2` (local) / `nvidia/nv-embedqa-e5-v5` (cloud) | 1024-dim |
| Reranker | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Optional, improves precision |
