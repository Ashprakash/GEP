#!/usr/bin/env bash
# Download a MULTI-FAMILY, Mac-runnable model set + FinanceBench into ./.hf_cache.
# All run in fp16 on Apple Silicon (M4 Max / 64 GB) — no CUDA, no bitsandbytes, no 4-bit.
# Models load ONE AT A TIME per run, so RAM is fine; disk for the full set is ~50 GB.
# Run on a networked machine with Hugging Face access.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/.hf_cache"

echo "==> FinanceBench dataset"
huggingface-cli download PatronusAI/financebench --repo-type dataset

# --- OPEN models (no gating) — the core cross-family comparison ---
echo "==> Qwen2.5 (0.5B dev + 7B main)"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

echo "==> DeepSeek (deepseek-llm-7b-chat)"
huggingface-cli download deepseek-ai/deepseek-llm-7b-chat

echo "==> Nemotron (NVIDIA Nemotron-Mini-4B-Instruct)"
huggingface-cli download nvidia/Nemotron-Mini-4B-Instruct

echo "==> Phi-3.5-mini (Microsoft, open)"
huggingface-cli download microsoft/Phi-3.5-mini-instruct

# --- GATED models (accept license + `huggingface-cli login` first) — optional extras ---
# huggingface-cli download google/gemma-2-9b-it
# huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

echo
echo "Done. Cache: $PWD/.hf_cache"
