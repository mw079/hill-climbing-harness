"""Build the guard split: three ~1,500-token prompts (a length no other split uses), 32 new tokens.

Checked pass/fail only (the same correctness rule as every split), never scored, never shown
to agents. Prompts come from the front of the shuffled GSM8K test pool; the test split's
prompts were taken from its end, so they do not overlap. Run like prepare_data.py.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
import prepare_data as prep  # noqa: E402

PROMPT_TOKENS, NEW_TOKENS, COUNT = 1500, 32, 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gsm8k", type=Path, required=True)
    p.add_argument("--out", type=Path, default=TASK / "data", help="where the data folder is")
    args = p.parse_args()
    sys.path.insert(0, str(TASK / "reference"))
    from engine import Engine
    engine = Engine(TASK / prep.POLICY["model_dir"])
    tok = engine.tokenizer
    rng = random.Random(prep.SEED)                        # the same shuffles as prepare_data.py
    train_rows = pq.read_table(next(args.gsm8k.glob("train-*.parquet"))).to_pylist()
    test_rows = pq.read_table(next(args.gsm8k.glob("test-*.parquet"))).to_pylist()
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    rows = list(reversed(test_rows[:600]))                # popped from the front of the pool
    prep.GROUPS["guard"] = (PROMPT_TOKENS, NEW_TOKENS)
    warmup = [{"prompt": prep.build_prompt(tok, rows, "short"), "new_tokens": 16},
              {"prompt": prep.build_prompt(tok, rows, "medium"), "new_tokens": 4}]
    requests, truth = [], []
    for _ in range(COUNT):
        prompt = prep.build_prompt(tok, rows, "guard")
        tokens, ids, lps = [], [], []
        for token, logprobs in engine.steps(prompt, NEW_TOKENS):
            lp = logprobs.astype(mx.float32)
            top = mx.argsort(-lp)[:prep.POLICY["top_k"]]
            tokens.append(int(token))
            ids.append(top.tolist())
            lps.append([round(x, 5) for x in lp[top].tolist()])
        requests.append({"group": "guard", "prompt": prompt, "new_tokens": NEW_TOKENS})
        truth.append({"tokens": tokens, "topk_ids": ids, "topk_logprobs": lps})
        print("guard", len(prompt), "prompt tokens", flush=True)
    out = args.out / "guard"
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.json").write_text(json.dumps({"warmup": warmup, "requests": requests}) + "\n")
    (out / "truth.json").write_text(json.dumps({"requests": truth}) + "\n")


if __name__ == "__main__":
    main()
