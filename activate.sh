# Source this (do NOT execute) to work inside the isolated GEP environment:
#   source activate.sh
#
# Activates ./.venv and pins ALL Hugging Face model/dataset downloads to
# ./.hf_cache so the experiment stays fully contained in this folder.

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Activate the local venv if present.
if [ -f "$_here/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$_here/.venv/bin/activate"
else
  echo "No .venv yet — run: bash setup_env.sh" >&2
fi

# Keep every download inside ./.hf_cache (not ~/.cache/huggingface).
export HF_HOME="$_here/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$_here/.hf_cache/hub"
export HF_DATASETS_CACHE="$_here/.hf_cache/datasets"

echo "GEP env active | HF_HOME=$HF_HOME"
