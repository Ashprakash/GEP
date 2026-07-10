#!/usr/bin/env bash
# Set up an isolated environment for the GEP pilot, entirely inside this
# folder. Creates ./.venv and keeps all model/dataset downloads in ./.hf_cache,
# so nothing is installed system-wide and nothing leaks outside the repo.
#
# Usage:
#   bash setup_env.sh          # create venv + install deps
#   source activate.sh         # then activate + point caches here for each shell
set -euo pipefail

cd "$(dirname "$0")"   # repo root

echo "==> Creating isolated venv at ./.venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip and installing requirements-colab.txt"
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt

echo
echo "Done. For each new shell, run:  source activate.sh"
echo "Then get the numbers with:      python run_pilot.py --stages baseline --n 5"
