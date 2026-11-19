"""Evaluator-owned timing driver. Runs in the candidate's environment.

    python driver.py <candidate_dir> <model_dir> <workload.json> <out.json>

Loads the candidate's Engine (untimed), runs the warm-up requests (untimed), then times
each request's generate() call on its own, one after another. Writes token ids, per-request
seconds and peak MLX memory. Correctness is judged by the evaluator, not here.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path


def main():
    candidate, model_dir, workload, out = map(Path, sys.argv[1:5])
    sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location("candidate_engine", candidate / "engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import mlx.core as mx

    work = json.loads(workload.read_text())
    started = time.perf_counter()
    engine = module.Engine(str(model_dir))
    load_seconds = time.perf_counter() - started
    for w in work["warmup"]:
        engine.generate(list(w["prompt"]), int(w["new_tokens"]))

    results = []
    for r in work["requests"]:
        prompt, n = list(r["prompt"]), int(r["new_tokens"])
        t = time.perf_counter()
        tokens = engine.generate(prompt, n)
        seconds = time.perf_counter() - t
        if not isinstance(tokens, list) or len(tokens) != n or not all(type(x) is int for x in tokens):
            raise SystemExit("generate must return a list of exactly max_new_tokens Python ints")
        results.append({"tokens": tokens, "seconds": seconds})
    out.write_text(json.dumps({"load_seconds": load_seconds, "results": results,
                               "peak_memory_gb": mx.get_peak_memory() / 2**30}))


if __name__ == "__main__":
    main()
