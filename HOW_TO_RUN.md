# GEP — Running GEP and Getting the Numbers

`GEP/` is a clone of [Ashprakash/GEP](https://github.com/Ashprakash/GEP) plus a
small terminal harness so the pilot can run **outside Colab** and stay **fully isolated inside
this folder** (own venv, own HuggingFace cache, own results).

GEP tests whether small models decide from grounded financial evidence rather than
memorized knowledge. The "numbers" are the pilot tables: `question_only` vs `with_gold_evidence`
accuracy, the grounding probe, and the template-reliability comparison (Brier / ECE / refusal).

---

## Can this run without network access?

**Only the offline mimic.** Real model runs need PyPI/Hugging Face access to install
`torch`/`transformers`/`datasets`, download the reader/extractor models, and pull the
FinanceBench dataset. Run the real steps below on **Colab (recommended)** or any
**networked machine**.

Numbers already recorded from prior Colab runs live in
[`benchmark/results_log.md`](benchmark/results_log.md).

---

## Reaching 75%+ — the headline path (selective accuracy on compact bundles)

Full-coverage accuracy on raw FinanceBench is low for a small model (~0.2 in `results_log.md`).
The repo's own route to a defensible 75%+ number (see the selective-prediction section in
`benchmark/financebench_pilot.py` and notebook cell 7c) stacks two levers:

1. **Extraction cascade** — the **7B extractor** turns noisy raw evidence into a compact bundle
   **plus a support probability**, without ever seeing the gold answer. The reader answers from
   the bundle. This lifts accuracy *and* yields a confidence signal that tracks correctness
   (the small model's own verbalized confidence doesn't).
2. **Selective prediction** — answer only the high-support cases, abstain on the rest. Reported
   as **accuracy@coverage**: `acc@30` / `acc@50` clear 75% even when `acc@100` is modest. This is
   an honest, standard reliability metric, not answer-copying.

See the mechanism offline right now (illustrative synthetic numbers):

```bash
python3 mock_pilot.py --stages selective
# acc@100 ≈ 0.45  ->  acc@50 ≈ 0.80  ->  acc@30 ≈ 1.0
```

Run it for real on your Mac (7B extractor + reader, MPS fp16):

```bash
source activate.sh
python run_pilot.py --stages cascade --n 30 \
  --extractor Qwen/Qwen2.5-7B-Instruct \
  --model    Qwen/Qwen2.5-0.5B-Instruct \
  --device auto
```

This prints the `EXTRACTION CASCADE SUMMARY` (accuracy + calibration: verbalized vs inherited
ECE, support/correctness correlation) and the `SELECTIVE ACCURACY` table (`acc@30`/`acc@50`),
and writes `results/cascade_*.csv`. To push the headline higher, also try the 7B as the *reader*
(`--model Qwen/Qwen2.5-7B-Instruct`): large-model bundle accuracy at reduced coverage is the
strongest honest number.

**Honesty note:** 75% is reported as "accuracy at N% coverage" (answering the confident subset),
or as the large-model/bundle reference — not as full-coverage accuracy of a 0.5B on raw evidence.
The actual figure depends on the real run; the mechanism (calibrated support → steep
risk-coverage curve) is what makes it reachable.

---

---

## Option 0 — Offline mimic (runs right now, on this Mac, no deps)

`mock_pilot.py` reproduces the pilot pipeline with pure stdlib: a synthetic
FinanceBench-like sample → a stubbed `MockModel` → the **real scoring functions**
(copied verbatim from `benchmark/financebench_pilot.py`) → the same summary tables.
It's tuned to mirror `results_log.md` (gold evidence doesn't rescue a tiny model and
worsens calibration; the model copies direct/counterfactual/compact evidence but fails
raw evidence; it mostly abstains on missing evidence).

```bash
cd GEP
python3 mock_pilot.py                 # baseline + probe, writes results/mock_*.csv
python3 mock_pilot.py --stages baseline
```

No network, no `pip`, no GPU. Use it to exercise the pipeline/metrics and demo the
findings. To get **real** numbers, swap `MockModel` for `pilot.load_hf_generator(...)`
— i.e. run `run_pilot.py` (Option A/B below) on a networked machine.

---

## Models for Mac (no CUDA) — testing the hypothesis, generating numbers

The pilot stages (`baseline` / `probe` / `template`) load the model in plain fp16/fp32 —
**they never use bitsandbytes/CUDA 4-bit** (only the LoRA training cells 27–37 do). So the
numbers that test the GEP hypothesis run entirely on Apple Silicon.

**Target hardware here: M4 Max / 64 GB unified memory.** That runs big models in fp16 on MPS
(rule of thumb ≈ 2 GB per 1B params), so we don't settle for tiny models. **Use multiple model
families** so the result isn't a Qwen artifact (reviewers will ask). All of these run locally in
fp16 — no CUDA/bitsandbytes — and load one at a time, so 64 GB RAM is plenty:

| Model | Family | fp16 RAM | Gating |
|-------|--------|----------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | Qwen | ~14 GB | open |
| `deepseek-ai/deepseek-llm-7b-chat` | DeepSeek | ~14 GB | open |
| `nvidia/Nemotron-Mini-4B-Instruct` | Nemotron | ~8 GB | open |
| `microsoft/Phi-3.5-mini-instruct` | Phi | ~8 GB | open |
| `Qwen/Qwen2.5-0.5B-Instruct` | Qwen (dev/scale) | ~1 GB | open |
| `google/gemma-2-9b-it` · `mistralai/Mistral-7B-Instruct-v0.3` · `meta-llama/Llama-3.1-8B-Instruct` | Gemma/Mistral/Llama | ~14–18 GB | **gated** (accept + `huggingface-cli login`) |

Four **open** families (Qwen, DeepSeek, Nemotron, Phi) give a friction-free cross-family
comparison; the gated three are optional extras. Chat templates are handled automatically, so
each is a drop-in `--model` / `--extractor`.

> Caveat: use plain *instruct/chat* models, not reasoning models (e.g. DeepSeek-R1 distills emit
> long `<think>` traces that crowd out the JSON and tank the executed rate). `deepseek-llm-7b-chat`
> is the right DeepSeek for structured extraction.

Download the multi-family set into the isolated cache (your own Mac terminal):

```bash
cd GEP
bash download_models.sh      # FinanceBench + Qwen(0.5B,7B) + DeepSeek + Nemotron + Phi (~50 GB disk)
```

Run the pilot across all families in **one resilient command** (checkpoints after every example
and every model, so a crash never loses finished work — just re-run to resume):

```bash
source activate.sh
python run_all.py --n 50               # default: Qwen-7B, DeepSeek, Nemotron, Phi
# resume after any crash: re-run the exact same command (done models/examples are skipped)
# recompute everything: add --force
```

This writes per-model results to `results/by_model/<label>.csv` (+ resumable `.jsonl`), a
`results/run_all_manifest.csv` (OK/FAILED per model), and regenerates the combined
`results/multimodel.{csv,md,tex}` after **every** model — so you always have a paste-ready
multi-model table even from a partial sweep. Drop `multimodel.tex` into the paper's Table 5.

For a single model or the per-stage breakdown, use `run_pilot.py` directly:

```bash
python run_pilot.py --stages tool --n 50 --extractor microsoft/Phi-3.5-mini-instruct --device auto --max-new-tokens 256
```

> Speed: transformers on MPS works but isn't the fastest path on Apple Silicon. If 7B generation
> feels slow, MLX (`mlx-lm`) or a GGUF/llama.cpp build of the same model are much faster — a
> future swap for the generator, not needed to get the numbers.

---



Open the runner notebook and pick a T4 GPU (`Runtime → Change runtime type → T4 GPU`):

<https://colab.research.google.com/github/Ashprakash/GEP/blob/main/benchmark/groundfin_colab_runner.ipynb>

Run the cells top to bottom. The numbers print as `=== HF SUMMARY ===`, `=== PROBE SUMMARY ===`,
`=== TEMPLATE SUMMARY ===` and are written to CSVs. Cell 13 is the Stage-1 baseline; cells 15/17
are the probe and template comparison.

---

## Option B — Local / any networked machine, isolated in this folder

Everything below stays inside `GEP/` — a `.venv/` here, downloads in `.hf_cache/` here,
outputs in `results/` here. Nothing is installed system-wide.

```bash
cd GEP

# 1. One-time: create the isolated venv and install deps into it
bash setup_env.sh

# 2. Each shell: activate the venv + pin HF caches to ./.hf_cache
source activate.sh

# 3. Get the numbers (tiny smoke run first)
python run_pilot.py --stages baseline --n 5
```

Then scale up / run the other stages:

```bash
python run_pilot.py --stages all --n 20                      # baseline + probe + template
python run_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 50
```

`run_pilot.py` is a thin CLI over the tested logic in `benchmark/financebench_pilot.py`
(the same functions the Colab cells call). It prints each summary table and writes
`results/<stage>_summary.csv` + `results/<stage>_results.csv`.

**Hardware note:** the default `Qwen/Qwen2.5-0.5B-Instruct` at small `n` runs on CPU (minutes,
slow). For `n ≥ 20` or larger models, use a CUDA GPU. First run downloads the model (~1 GB for
0.5B) and the FinanceBench dataset into `./.hf_cache`.

> On this Mac (Apple Silicon): `run_pilot.py` will run the real model on **CPU** — the repo's
> `load_hf_generator` only picks fp16/CUDA when `torch.cuda.is_available()`, so it falls back to
> fp32/CPU here (MPS isn't auto-selected). Fine for the 0.5B model at small `n`; just slower.
> The install itself (`bash setup_env.sh`) must run in your own Mac terminal, where PyPI is
> reachable.

---

## What each stage reports

| Stage | Command | Question it answers |
|-------|---------|---------------------|
| `baseline` | `--stages baseline` | Does gold evidence beat question-only? (weak/numeric accuracy, refusal, Brier, ECE) |
| `probe` | `--stages probe` | Can the model follow gold / compressed / direct / counterfactual evidence, and abstain when it's missing? |
| `template` | `--stages template` | Do risk-calibrated templates improve reliability vs raw evidence / summaries / traces? |

The deeper method steps (scaled suite, LoRA SFT, teacher distillation, GRPO) live in the
notebook cells 7b+ and `benchmark/train_groundfin.py`; those need a real GPU and are beyond the
"get the pilot numbers" scope here.

---

## Isolation summary (per your request)

- **Code**: cloned into `GEP/` (this folder), with its own `.git` for `git pull`.
- **Python env**: `GEP/.venv/` — created by `setup_env.sh`, never system-wide.
- **Downloads**: `GEP/.hf_cache/` — `activate.sh` sets `HF_HOME` here, so models/datasets
  never land in `~/.cache/huggingface`.
- **Outputs**: `GEP/results/*.csv`.
- All of the above are git-ignored (see `.gitignore`) so they won't be committed.
