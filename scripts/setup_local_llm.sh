#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${SALES_AI_LOCAL_MODEL:-qwen3:4b-instruct}"

if command -v ollama >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama)"
elif [[ -x /Applications/Ollama.app/Contents/Resources/ollama ]]; then
  OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
else
  echo "Ollama is not installed. Download the macOS app from https://ollama.com/download."
  exit 1
fi

echo "Downloading ${MODEL_NAME} for local inference..."
"${OLLAMA_BIN}" pull "${MODEL_NAME}"
echo "Local model ready: ${MODEL_NAME}"
