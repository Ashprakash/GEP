# Reproducing GEP

Everything needed to pull the code and reproduce the **4 models × 3 benchmarks** evaluation
(baseline vs GEP, coverage, yield) and the confidence-signal study. Model weights are **not**
in the repo — you download them locally (kept in `./.hf_cache`, git-ignored).

## Requirements
- Python 3.10+
- Apple Silicon (MPS), CUDA GPU, or CPU. On a 64 GB Mac all models run in fp16 (no quantization).
- ~60 GB disk for the model set (downloaded once into `./.hf_cache`).
- Network for the first run (HuggingFace models/datasets + FinQA JSON from GitHub).

## Quickstart
```bash
git clone <your-fork-url> GEP && cd GEP
bash setup_env.sh          # creates ./.venv, installs requirements.txt
source activate.sh         # activates venv + pins HF_HOME=./.hf_cache
bash download_models.sh    # FinanceBench + Qwen(0.5B,7B) + DeepSeek + Nemotron + Phi (~50 GB)
```
Then the smallest end-to-end check (offline, no models, no network):
```bash
python3 mock_pilot.py      # sanity-checks the pipeline logic with a stub model
```

## Reproduce the master table (4 models × 3 benchmarks)
One line per benchmark (models cached after `download_models.sh`; `--with-baseline` also runs the
raw baseline; resumable — re-run to continue after any interruption):
```bash
python run_all.py --benchmark financebench --outdir results/financebench --with-baseline --n 100 --models "Qwen/Qwen2.5-7B-Instruct,deepseek-ai/deepseek-llm-7b-chat,nvidia/Nemotron-Mini-4B-Instruct,microsoft/Phi-3.5-mini-instruct"
python run_all.py --benchmark finqa        --outdir results/finqa        --with-baseline --n 100 --models "Qwen/Qwen2.5-7B-Instruct,deepseek-ai/deepseek-llm-7b-chat,nvidia/Nemotron-Mini-4B-Instruct,microsoft/Phi-3.5-mini-instruct"
python run_all.py --benchmark drop         --outdir results/drop         --with-baseline --n 100 --models "Qwen/Qwen2.5-7B-Instruct,deepseek-ai/deepseek-llm-7b-chat,nvidia/Nemotron-Mini-4B-Instruct,microsoft/Phi-3.5-mini-instruct"
```
> `make_table.py` reads `results/{financebench,finqa,drop}/by_model/`. If you keep FinanceBench in
> the default `results/by_model/`, edit the `DATASETS` paths at the top of `make_table.py`.

Then generate the tables and analyses:
```bash
python3 make_table.py                              # 12-cell table -> results/master_table.{md,tex}
python3 compare_selective.py results/finqa/by_model/*.csv    # confidence signals: AURC/ECE per model
python3 phase_transition.py  results/finqa/by_model/*.csv    # signal ranking vs model (crossover view)
python3 inspect_tool.py results/finqa/by_model/Qwen2.5-7B-Instruct.csv   # per-example audit
```

## LLM-as-judge (optional, fairer scorer)
```bash
export JUDGE_API_KEY=... JUDGE_MODEL=gpt-4o        # + JUDGE_BASE_URL for non-OpenAI
python llm_judge.py results/finqa/by_model/Qwen2.5-7B-Instruct.csv --pred-col computed
```

## File index
| File | Purpose |
|------|---------|
| `method_tool.py` | GEP: extract typed variables+formula → deterministic executor → grounding/verbalized/logprob confidence → abstain |
| `run_pilot.py` | Single-model runner (`--stages baseline,tool,probe,template,cascade`), device-aware MPS loader |
| `run_all.py` | Resilient multi-model sweep (per-example + per-model checkpoints, `--with-baseline`, `--benchmark`) |
| `benchmarks.py` | Dataset loaders → common schema: `financebench`, `finqa` (GitHub JSON), `drop` |
| `make_table.py` | 12-cell master table (baseline / acc@commit / coverage / yield) → md + tex |
| `compare_selective.py` | Confidence-signal risk–coverage: grounding vs verbalized vs log-prob vs ensemble (AURC/ECE) |
| `phase_transition.py` | Signal ranking as a function of model size (within-family crossover view) |
| `rescore.py` | Re-score saved runs with the current executor — no model re-run |
| `inspect_tool.py` | Per-example audit (question / gold / computed / formula / raw / fail reason) |
| `write_results.py` | Paste-ready md + LaTeX table bodies from a run |
| `llm_judge.py` | Reproducible LLM-as-judge scoring |
| `mock_pilot.py` | Offline, dependency-free pipeline sanity check |
| `setup_env.sh` / `activate.sh` / `download_models.sh` | Isolated venv + HF cache + model downloads |
| `RESULTS.md` / `HYPOTHESIS.md` / `EXPERIMENT_PLAN.md` / `METHOD_v2.md` | Findings, claim, plan, method notes |
| `paper/gep.tex` | KDD-style (ACM sigconf) draft |

## Notes & gotchas
- **Model cache** lives in `./.hf_cache` (git-ignored). `run_all.py`/`run_pilot.py` pin `HF_HOME` there
  automatically, so models aren't re-downloaded even if you forget `source activate.sh`.
- **FinQA** loads from its GitHub JSON (the HF version is a removed script dataset). If GitHub is
  blocked, download `dataset/test.json` from `github.com/czyssrs/FinQA` and set `FINQA_JSON=/path`.
- **DROP** uses `ucinlp/drop` (parquet). **Gemma/Llama/Mistral** are gated — `huggingface-cli login` first.
- **Decode is pure-greedy** — do not add repetition penalties (they corrupt JSON extraction).
- **Determinism**: pass `--seed`; for paper-grade numbers run ≥3 seeds and report CIs.
- Outputs (`results/`), venv (`.venv/`), and cache (`.hf_cache/`) are all git-ignored.

## What the results show (see RESULTS.md)
GEP's acc@commit beats the raw baseline in all 12 cells, but the coverage-fair `yield` wins
7/12 (it loses where it abstains too much). No single confidence signal transfers across
benchmarks; the grounding×verbalized ensemble is the best-calibrated. Report **both** acc@commit
and yield, and give the baseline its own abstention (matched-coverage) for a fair comparison.
