"""Select and freeze one FAISS setting using train only (all train datasets together).

The winner must meet the recall floor on every train dataset; among those, the largest
geometric mean of per-dataset median QPS wins. One setting is used for every dataset,
the same rule the candidate faces: one engine, no per-dataset hand tuning from hidden data.
"""
import importlib.util
import json
import math
import platform
import statistics
import tempfile
from pathlib import Path

TASK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("vector_evaluator", TASK / "evaluator/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    if (TASK / "check_report.json").exists():
        raise SystemExit("task already checked; reference tuning is frozen")
    settings = [("Flat", 0)]
    settings += [(name, ef) for name in ("HNSW16", "HNSW32") for ef in (32, 64, 128, 256)]
    settings += [("IVF64,Flat", nprobe) for nprobe in (8, 16, 32, 64)]
    train = ev.datasets(TASK / "data/train")
    points = []
    with tempfile.TemporaryDirectory(prefix="faiss-tune-") as tmp:
        tmp = Path(tmp)
        runner = tmp / "engine"
        ev.compile_runner(TASK / "reference", runner, reference=True, tuning=True)
        for factory, parameter in settings:
            env = {"HC_TUNE_FACTORY": factory, "HC_TUNE_PARAMETER": str(parameter)}
            per = {}
            for name, path in train:
                work = tmp / name
                work.mkdir(exist_ok=True)
                reps = [ev.run_runner(runner, path, work, env) for _ in range(3)]
                per[name] = {"qps": statistics.median(r["qps"] for r in reps),
                             "recall_at_10": min(r["recall_at_10"] for r in reps),
                             "max_build_seconds": max(r["build_seconds"] for r in reps),
                             "max_rss_mb": max(r["peak_rss_mb"] for r in reps)}
            point = {"factory": factory, "parameter": parameter,
                     "qps_geomean": math.exp(statistics.mean(math.log(p["qps"]) for p in per.values())),
                     "min_recall_at_10": min(p["recall_at_10"] for p in per.values()),
                     "datasets": per}
            points.append(point)
            print(json.dumps({k: v for k, v in point.items() if k != "datasets"}), flush=True)
    valid = [p for p in points if p["min_recall_at_10"] >= ev.POLICY["recall_floor"]]
    if not valid:
        raise SystemExit("no FAISS setting met the training constraints")
    winner = max(valid, key=lambda p: p["qps_geomean"])
    # No validation/test data are read by this script.
    (TASK / "reference/frozen.h").write_text('#pragma once\n// Frozen using train only.\n'
        f'constexpr const char* HC_FACTORY = "{winner["factory"]}";\n'
        f'constexpr int HC_SEARCH_PARAMETER = {winner["parameter"]};\n')
    (TASK / "reference/tuning.json").write_text(json.dumps({"selection_split": "train",
        "train_datasets": [n for n, _ in train],
        "selection_rule": "largest geometric-mean median QPS with recall floor met on every train dataset",
        "timing_repetitions": 3, "policy": ev.POLICY, "machine": platform.platform(),
        "settings": points, "selected": winner,
        "scope": "best tested setting in this finite catalog, not all possible FAISS indexes"}, indent=2) + "\n")
    print("Frozen training winner:", json.dumps({k: v for k, v in winner.items() if k != "datasets"}))


if __name__ == "__main__":
    main()
