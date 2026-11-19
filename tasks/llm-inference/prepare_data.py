"""Build prompt workloads and reference truth for each split.

Run once in an environment with mlx-lm and pyarrow (see README). Prompts are GSM8K
questions (MIT licence), tokenized here, so candidates never need a tokenizer.

- train, val: disjoint rows of GSM8K train; groups short (gen 128) and medium (~512-token
  few-shot prompts, gen 64).
- test: GSM8K test rows; groups short, medium and long (~2048-token prompts, gen 64).
  The long group appears only in test, to check that speed-ups generalise to long contexts.

truth.json holds mlx-lm's greedy tokens and top-k log-probabilities at every step.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parent
POLICY = json.loads((TASK / "policy.json").read_text())
SEED = 20261116
GROUPS = {"short": (0, 128), "medium": (512, 64), "long": (2048, 64)}   # prompt tokens, new tokens
PLAN = {"train": ["short", "medium"], "val": ["short", "medium"], "test": ["short", "medium", "long"]}
PER_GROUP = 4


def qa(row, answer=True):
    return f"Question: {row['question']}\nAnswer:" + (f" {row['answer']}\n\n" if answer else "")


def build_prompt(tok, rows, group):
    """A question, preceded by worked examples until the prompt reaches the group's size."""
    target = GROUPS[group][0]
    question = rows.pop()
    shots = ""
    while target and len(tok.encode(shots + qa(question, False))) < target:
        shots += qa(rows.pop())
    return tok.encode(shots + qa(question, False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gsm8k", type=Path, required=True, help="folder with train-*.parquet and test-*.parquet")
    p.add_argument("--out", type=Path, default=TASK / "data", help="where to write the splits")
    p.add_argument("--reproduce", action="store_true",
                   help="rebuild the recorded splits (setup.sh); allowed on a checked task, keeps preparation.json")
    args = p.parse_args()
    if (TASK / "check_report.json").exists() and not args.reproduce:
        p.error("task is already checked; use a new task copy for a new data protocol")
    sys.path.insert(0, str(TASK / "reference"))
    from engine import Engine
    engine = Engine(TASK / POLICY["model_dir"])
    tok = engine.tokenizer

    rng = random.Random(SEED)
    train_rows = pq.read_table(next(args.gsm8k.glob("train-*.parquet"))).to_pylist()
    test_rows = pq.read_table(next(args.gsm8k.glob("test-*.parquet"))).to_pylist()
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    half = len(train_rows) // 2
    pools = {"train": train_rows[:half], "val": train_rows[half:], "test": test_rows}

    provenance = {"seed": SEED, "source": "openai/gsm8k (MIT)", "groups": GROUPS, "plan": PLAN,
                  "per_group": PER_GROUP, "policy": POLICY, "splits": {}}
    for split, groups in PLAN.items():
        rows = list(pools[split])
        warmup = [{"prompt": build_prompt(tok, rows, "short"), "new_tokens": 16},
                  {"prompt": build_prompt(tok, rows, "medium"), "new_tokens": 4}]
        requests, truth = [], []
        for group in groups:
            for _ in range(PER_GROUP):
                prompt, n = build_prompt(tok, rows, group), GROUPS[group][1]
                tokens, ids, lps = [], [], []
                for token, logprobs in engine.steps(prompt, n):
                    lp = logprobs.astype(mx.float32)
                    top = mx.argsort(-lp)[:POLICY["top_k"]]
                    tokens.append(int(token))
                    ids.append(top.tolist())
                    lps.append([round(x, 5) for x in lp[top].tolist()])
                requests.append({"group": group, "prompt": prompt, "new_tokens": n})
                truth.append({"tokens": tokens, "topk_ids": ids, "topk_logprobs": lps})
                print(split, group, len(prompt), "prompt tokens", flush=True)
        out = args.out / split
        out.mkdir(parents=True, exist_ok=True)
        (out / "workload.json").write_text(json.dumps({"warmup": warmup, "requests": requests}) + "\n")
        (out / "truth.json").write_text(json.dumps({"requests": truth}) + "\n")
        provenance["splits"][split] = {
            "prompt_tokens": [len(r["prompt"]) for r in requests],
            "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())}}
    if not args.reproduce:
        (TASK / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
