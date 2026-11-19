"""Independent vector-search scorer over one or more datasets per split.
Only aggregate results leave this process."""
import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1]
POLICY = json.loads((TASK / "policy.json").read_text())
FLAGS = ["-O3", "-mcpu=native", "-DNDEBUG", "-std=c++20"]


def matrix(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype="<u4", count=2)
        if len(header) != 2:
            raise ValueError("invalid matrix header")
        n, d = map(int, header)
        data = np.fromfile(f, dtype="<f4")
    if not n or not d or len(data) != n * d or not np.isfinite(data).all():
        raise ValueError("invalid matrix payload")
    return data.reshape(n, d)


def score_ids(ids, base, queries, kth):
    k = POLICY["k"]
    if ids.shape != (len(queries), k) or ids.dtype.kind not in "iu":
        raise ValueError("invalid result shape or type")
    if np.any(ids < 0) or np.any(ids >= len(base)):
        raise ValueError("out-of-range result ID")
    if np.any(np.diff(np.sort(ids.astype(np.int64), axis=1), axis=1) == 0):
        raise ValueError("duplicate result IDs")
    if kth.shape != (len(queries),) or not np.isfinite(kth).all() or np.any(kth < 0):
        raise ValueError("invalid ground truth")
    diff = base[ids].astype(np.float64) - queries[:, None, :].astype(np.float64)
    distances = np.einsum("qkd,qkd->qk", diff, diff)
    tolerance = POLICY["distance_tolerance"] * np.maximum(1.0, kth)
    return float(np.mean(distances <= kth[:, None] + tolerance[:, None]))


def compile_runner(candidate, output, reference=False, tuning=False):
    command = ["/usr/bin/clang++", *FLAGS, "-I", str(candidate),
               str(TASK / "evaluator/driver.cpp"), "-framework", "Accelerate"]
    if reference:
        vendor = TASK / "reference/vendor"
        command += ["-I", str(vendor / "include"), "-I/opt/homebrew/opt/libomp/include",
                    "-Xpreprocessor", "-fopenmp", str(vendor / "libfaiss.a"),
                    "-L/opt/homebrew/opt/libomp/lib", "-lomp"]
    if tuning:
        command += ["-DHC_TUNING"]
    command += ["-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        # Compiler diagnostics can contain source or paths, so expose no raw text.
        raise ValueError("candidate did not compile with the official driver")


def run_runner(executable, split, temporary, extra_env=None):
    base_path, query_path = temporary / "base.bin", temporary / "queries.bin"
    # Child receives only vector inputs; neither truth nor original split paths.
    shutil.copyfile(split / "base.bin", base_path)
    shutil.copyfile(split / "queries.bin", query_path)
    output, metrics_file = temporary / "ids.bin", temporary / "metrics.json"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(extra_env or {})}
    started = time.perf_counter()
    try:
        result = subprocess.run([str(executable), str(base_path), str(query_path), str(output),
                                 str(metrics_file), str(POLICY["min_search_seconds"]),
                                 str(POLICY["max_build_seconds"]), str(POLICY["max_rss_mb"])],
                                cwd=temporary, env=env, capture_output=True,
                                timeout=POLICY["process_timeout_seconds"])
    except subprocess.TimeoutExpired:
        raise ValueError("candidate process exceeded its time limit") from None
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise ValueError("candidate process failed correctness, resource, or runtime checks")
    metrics = json.loads(metrics_file.read_text())
    keys = ("build_seconds", "search_seconds", "passes", "queries", "peak_rss_mb")
    if any(isinstance(metrics.get(k), bool) or not isinstance(metrics.get(k), (int, float))
           or not math.isfinite(metrics[k]) or metrics[k] < 0 for k in keys):
        raise ValueError("invalid timing or resource metrics")
    if (metrics["passes"] < 1 or int(metrics["passes"]) != metrics["passes"]
            or metrics["search_seconds"] < POLICY["min_search_seconds"]
            or metrics["search_seconds"] > elapsed + 0.01
            or metrics["build_seconds"] > POLICY["max_build_seconds"]
            or metrics["peak_rss_mb"] > POLICY["max_rss_mb"]):
        raise ValueError("timing or resource limits violated")
    base, queries = matrix(split / "base.bin"), matrix(split / "queries.bin")
    if output.stat().st_size != len(queries) * POLICY["k"] * 4 or metrics["queries"] != len(queries):
        raise ValueError("incorrect number of returned IDs")
    ids = np.fromfile(output, dtype="<u4").reshape(len(queries), POLICY["k"])
    recall = score_ids(ids, base, queries, np.load(split / "kth.npy", allow_pickle=False))
    qps = metrics["passes"] * len(queries) / metrics["search_seconds"]
    if not math.isfinite(qps) or qps <= 0:
        raise ValueError("invalid throughput")
    metrics.update(recall_at_10=recall, qps=qps)
    return metrics


def datasets(split):
    """A split is one dataset (base.bin at its top) or a folder of dataset subfolders."""
    if (split / "base.bin").exists():
        return [("data", split)]
    found = sorted((d.name, d) for d in split.iterdir() if (d / "base.bin").exists())
    if not found:
        raise ValueError("split holds no dataset")
    return found


def combine(per_dataset):
    """Score = geometric mean of per-dataset QPS, so each dataset counts equally
    regardless of its dimension. Valid only if every dataset meets the recall floor."""
    qps = [m["qps"] for m in per_dataset.values()]
    metrics = {"qps_geomean": math.exp(sum(math.log(q) for q in qps) / len(qps)),
               "min_recall_at_10": min(m["recall_at_10"] for m in per_dataset.values()),
               "max_build_seconds": max(m["build_seconds"] for m in per_dataset.values()),
               "max_peak_rss_mb": max(m["peak_rss_mb"] for m in per_dataset.values())}
    for name, m in per_dataset.items():
        metrics.update({f"{name}_qps": m["qps"], f"{name}_recall_at_10": m["recall_at_10"],
                        f"{name}_build_seconds": m["build_seconds"]})
    return metrics


def evaluate(candidate, split, reference=False):
    result = {"valid": False, "score": None, "metrics": {}, "error": None}
    try:
        with tempfile.TemporaryDirectory(prefix="vector-eval-") as tmp:
            tmp = Path(tmp)
            executable = tmp / "engine"
            compile_runner(candidate, executable, reference)
            per_dataset = {}
            for name, path in datasets(split):
                work = tmp / name
                work.mkdir()
                per_dataset[name] = run_runner(executable, path, work)
        metrics = combine(per_dataset)
        result["metrics"] = metrics
        low = [n for n, m in per_dataset.items() if m["recall_at_10"] < POLICY["recall_floor"]]
        if low:
            result["error"] = f"recall@10 is below the required floor on {len(low)} of {len(per_dataset)} dataset(s)"
        else:
            result.update(valid=True, score=metrics["qps_geomean"])
    except (ValueError, OSError, subprocess.SubprocessError):
        # Do not forward exception text: child-controlled output or file paths may
        # otherwise expose hidden examples. Aggregates above are the only feedback.
        result["error"] = "candidate failed official compilation, output, resource, or runtime checks"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.candidate.resolve(), args.split.resolve(),
                      reference=os.environ.get("HILLCLIMB_EVAL_KIND") == "ref")
    args.out.write_text(json.dumps(result) + "\n")
