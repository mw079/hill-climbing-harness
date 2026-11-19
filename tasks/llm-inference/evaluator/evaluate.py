"""Scorer for local LLM inference. Only aggregate results leave this process.

Score: geometric mean over request groups (short, medium, ...) of generated tokens per
second, where each group's rate is its total new tokens over its total generate() time.

Validity, two checks:
1. Against mlx-lm's own greedy outputs (truth.json, top-k log-probabilities per step): while
   a candidate follows them, each token must be the reference token or within
   `logprob_tolerance` of the best (a near-tie, which bf16 rounding can flip). After a
   near-tie the continuations legitimately differ.
2. Every generated token, including after a divergence: an evaluator-owned copy of the
   starting engine (verifier_engine.py) re-scores the candidate's output given its prompt and
   the candidate's own earlier tokens. Each token must be within `verifier_tolerance` of that
   model's best log-probability. Nothing a candidate generates goes unchecked.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
POLICY = json.loads((TASK / "policy.json").read_text())
MODEL = TASK / POLICY["model_dir"]


def check_tokens(tokens, truth):
    """Return (checked, diverged, error) for one request."""
    tol = POLICY["logprob_tolerance"]
    for i, (token, ref) in enumerate(zip(tokens, truth["tokens"])):
        if token == ref:
            continue
        ranked = dict(zip(truth["topk_ids"][i], truth["topk_logprobs"][i]))
        if token in ranked and truth["topk_logprobs"][i][0] - ranked[token] <= tol:
            return i + 1, True, None
        return i + 1, True, "a generated token is not a greedy choice of the reference model"
    return len(tokens), False, None


def score(work, truth, driver_out, elapsed, margins=None):
    results = driver_out["results"]
    if len(results) != len(work["requests"]):
        raise ValueError("wrong number of results")
    if sum(r["seconds"] for r in results) > elapsed + 0.05:
        raise ValueError("reported time exceeds the process wall time")
    groups, checked, total, diverged = {}, 0, 0, 0
    worst = 0.0
    for i, (req, res, ref) in enumerate(zip(work["requests"], results, truth["requests"])):
        n, s = req["new_tokens"], res["seconds"]
        if not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0):
            raise ValueError("invalid timing")
        c, d, error = check_tokens(res["tokens"], ref)
        if error:
            return None, {}, error
        if margins is not None:
            worst = max([worst] + margins[i])
            if any(m > POLICY["verifier_tolerance"] for m in margins[i]):
                return None, {}, "a generated token is not a greedy choice of the model"
        checked, total, diverged = checked + c, total + n, diverged + d
        g = groups.setdefault(req["group"], [0, 0.0])
        g[0] += n
        g[1] += s
    rates = {name: tokens / seconds for name, (tokens, seconds) in groups.items()}
    metrics = {"tokens_per_second_geomean": math.exp(sum(map(math.log, rates.values())) / len(rates)),
               **{f"{name}_tokens_per_second": r for name, r in rates.items()},
               "followed_reference_fraction": checked / total, "diverged_requests": diverged,
               "max_verifier_margin": worst,
               "requests": len(results), "load_seconds": driver_out["load_seconds"],
               "peak_memory_gb": driver_out["peak_memory_gb"]}
    if metrics["peak_memory_gb"] > POLICY["max_peak_memory_gb"]:
        return None, metrics, "peak memory limit exceeded"
    return metrics["tokens_per_second_geomean"], metrics, None


def evaluate(candidate, split):
    result = {"valid": False, "score": None, "metrics": {}, "error": None}
    try:
        work = json.loads((split / "workload.json").read_text())
        truth = json.loads((split / "truth.json").read_text())
        with tempfile.TemporaryDirectory(prefix="llm-eval-") as tmp:
            tmp = Path(tmp)
            # The candidate process receives prompts only, never the truth file or split path.
            shutil.copyfile(split / "workload.json", tmp / "workload.json")
            started = time.perf_counter()
            proc = subprocess.run([sys.executable, str(Path(__file__).parent / "driver.py"), str(candidate),
                                   str(MODEL), str(tmp / "workload.json"), str(tmp / "out.json")],
                                  cwd=tmp, capture_output=True, text=True,
                                  timeout=POLICY["process_timeout_seconds"],
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            elapsed = time.perf_counter() - started
            if proc.returncode:
                raise ValueError("candidate process failed")
            driver_out = json.loads((tmp / "out.json").read_text())
        sys.path.insert(0, str(Path(__file__).parent))
        from verifier_engine import Engine as Verifier
        verifier = Verifier(str(MODEL))
        margins = [verifier.margins(req["prompt"], res["tokens"])
                   for req, res in zip(work["requests"], driver_out["results"])]
        value, metrics, error = score(work, truth, driver_out, elapsed, margins)
        result.update(metrics=metrics, error=error)
        if error is None:
            result.update(valid=True, score=value)
    except subprocess.TimeoutExpired:
        result["error"] = "candidate process exceeded its time limit"
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        # No raw text: child output could otherwise carry hidden prompts or paths.
        result["error"] = "candidate failed to load, run, or return well-formed output"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(evaluate(args.candidate.resolve(), args.split.resolve())) + "\n")
