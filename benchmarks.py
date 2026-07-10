"""
benchmarks.py — dataset loaders returning the common schema the pilot expects:
  columns: financebench_id, question, evidence_text, answer, has_numeric_answer,
           company, question_type, question_reasoning

Lets the tool method run on multiple benchmarks (generality for the paper):
  financebench  — anchor (delegates to the repo loader)
  finqa         — financial numeric reasoning w/ gold programs (in-domain depth)
  drop          — discrete numeric reasoning over general text (cross-domain)

NOTE: HF dataset field names vary by mirror/version. SMOKE-TEST each loader at n=3
before a full run and adjust field names if a KeyError appears.
"""
import re

_NUM = re.compile(r"-?\d")


def _has_num(a):
    return bool(_NUM.search(str(a)))


def _load_ds(candidates, split):
    """Try dataset id candidates in order (HF mirrors/renames vary)."""
    from datasets import load_dataset
    last = None
    for name in candidates:
        try:
            return load_dataset(name, split=split)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Could not load any of {candidates}: {last}")


def load_finqa(split="test"):
    """Load FinQA from its GitHub JSON (avoids the datasets-v3 no-scripts problem).
    Set FINQA_JSON=/path/to/test.json to load from a local file instead."""
    import json
    import os
    import urllib.request
    import pandas as pd

    local = os.getenv("FINQA_JSON")
    data = None
    if local and os.path.exists(local):
        with open(local) as f:
            data = json.load(f)
    else:
        urls = [f"https://raw.githubusercontent.com/czyssrs/FinQA/master/dataset/{split}.json",
                f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"]
        err = None
        for u in urls:
            try:
                with urllib.request.urlopen(u, timeout=60) as resp:
                    data = json.load(resp)
                    break
            except Exception as e:  # noqa: BLE001
                err = e
        if data is None:
            raise RuntimeError(
                f"Could not fetch FinQA {split}.json ({err}). Download it from "
                "github.com/czyssrs/FinQA (dataset/test.json) and set FINQA_JSON=/path.")

    rows = []
    for i, it in enumerate(data):
        qa = it.get("qa") or it.get("qa_0") or {}
        pre = " ".join(it.get("pre_text") or [])
        post = " ".join(it.get("post_text") or [])
        table = it.get("table") or []
        tbl = "\n".join(" | ".join(str(c) for c in trow) for trow in table)
        ev = f"{pre}\n\n{tbl}\n\n{post}".strip()
        ans = str(qa.get("exe_ans", qa.get("answer", "")))
        rid = str(it.get("id", f"finqa_{i}"))
        rows.append({"financebench_id": rid, "question": str(qa.get("question", "")),
                     "evidence_text": ev, "answer": ans,
                     "company": rid.split("/")[0] if "/" in rid else "finqa",
                     "question_type": "", "question_reasoning": ""})
    df = pd.DataFrame(rows)
    df["has_numeric_answer"] = df["answer"].apply(_has_num)
    return df


def load_drop(split="validation"):
    import pandas as pd
    ds = _load_ds(["ucinlp/drop", "drop"], split)
    rows = []
    for i, r in enumerate(ds):
        spans = (r.get("answers_spans") or {}).get("spans") or []
        ans = spans[0] if spans else ""
        rows.append({"financebench_id": str(r.get("query_id", f"drop_{i}")),
                     "question": str(r.get("question", "")),
                     "evidence_text": str(r.get("passage", "")), "answer": str(ans),
                     "company": "drop", "question_type": "", "question_reasoning": ""})
    df = pd.DataFrame(rows)
    df["has_numeric_answer"] = df["answer"].apply(_has_num)
    return df[df["has_numeric_answer"]].reset_index(drop=True)  # numeric subset


def get_benchmark(name):
    name = name.lower()
    if name == "financebench":
        from benchmark import financebench_pilot as pilot
        return pilot.load_financebench()
    if name == "finqa":
        return load_finqa()
    if name == "drop":
        return load_drop()
    raise ValueError(f"Unknown benchmark {name!r}; use financebench|finqa|drop")
