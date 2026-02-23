#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "  NVIDIA AI Workbench — RAG Setup Script"
echo "=============================================="
echo ""

# Check if AI Workbench is installed
if ! command -v nvwb &>/dev/null; then
    echo "[INFO] NVIDIA AI Workbench not detected."
    echo ""
    echo "To install on Ubuntu 22.04/24.04:"
    echo "  curl -fsSL https://workbench.download.nvidia.com/stable/linux/gpgkey | sudo tee -a /etc/apt/trusted.gpg.d/ai-workbench-desktop-key.asc"
    echo '  echo "deb https://workbench.download.nvidia.com/stable/linux/debian default proprietary" | sudo tee -a /etc/apt/sources.list'
    echo "  sudo apt update && sudo apt install nvidia-ai-workbench"
    echo ""
    echo "After installation, run this script again."
    echo ""
    read -p "Continue with manual Docker-based setup instead? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# NGC API Key
if [ -z "${NGC_API_KEY:-}" ]; then
    echo "[SETUP] NGC API Key required for NVIDIA NIM containers."
    echo "  Get one at: https://ngc.nvidia.com/signin"
    read -sp "Enter NGC API Key: " NGC_API_KEY
    echo
    export NGC_API_KEY
fi

# NVIDIA API Key for build.nvidia.com
if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "[SETUP] NVIDIA API Key for cloud endpoints (optional, press Enter to skip)."
    echo "  Get one at: https://build.nvidia.com"
    read -sp "Enter NVIDIA API Key (or press Enter): " NVIDIA_API_KEY
    echo
    export NVIDIA_API_KEY
fi

# Save to .env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

cat > "$ENV_FILE" <<EOF
NGC_API_KEY=${NGC_API_KEY}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-}
EOF
chmod 600 "$ENV_FILE"
echo "[OK] Saved credentials to .env"

# Check GPU
echo ""
echo "[INFO] GPU Status:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not found. GPU may not be available."
fi

# Launch NIM + Chatbot
echo ""
echo "[LAUNCH] Starting services..."
cd "$PROJECT_DIR"

echo "  Option 1: NIM + Chatbot (requires 24GB+ VRAM)"
echo "    docker compose --profile nim --profile app up -d"
echo ""
echo "  Option 2: Standalone (local model or cloud API)"
echo "    docker compose --profile standalone up -d"
echo ""
echo "  Option 3: Cloud-only (no GPU needed)"
echo "    RAG_MODE=nvidia_cloud python3 scripts/run.sh"
echo ""

read -p "Select option [1/2/3]: " -n 1 -r
echo
case $REPLY in
    1) docker compose --profile nim --profile app up -d ;;
    2) docker compose --profile standalone up -d ;;
    3) RAG_MODE=nvidia_cloud bash scripts/run.sh ;;
    *) echo "Invalid option" ;;
esac
