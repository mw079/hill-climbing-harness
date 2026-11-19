"""Tables and progress plots for the runs behind the README results.

    uv run --no-project --with matplotlib==3.10.8 python results/make_results.py [runs-dir]

Reads the four relevant run folders (default: runs/) and writes, next to this script:
<name>.tsv (one row per experiment), <name>.png (progress plot), vector-search-comparison.png,
and summary.json (the numbers in the README table). The run folders themselves are not kept
in git; these files are.
"""
import csv
import json
import tomllib
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = {   # name -> (run folder, harness, task, reference label)
    "vector-search-v1": ("vector-search-multi-20261117-004814", "v1 hillclimb.py", "vector-search-multi", "FAISS"),
    "vector-search-v2": ("vector-search-multi-multi-20261117-052206", "v2 hillclimb_multi.py", "vector-search-multi", "FAISS"),
    "vector-search-v3": ("vector-search-multi-agent-20261117-053848", "v3 hillclimb_agent.py", "vector-search-multi", "FAISS"),
    "llm-inference-v3": ("llm-inference-agent-20261117-153745", "v3 hillclimb_agent.py", "llm-inference", "mlx-lm"),
}
COLORS = {"vector-search-v1": "#1f77b4", "vector-search-v2": "#ff7f0e", "vector-search-v3": "#2ca02c"}


def short(title):
    t = re.sub(r"^Experiment \d+:\s*", "", title)
    t = re.sub(r"^\[[^\]]*\]\s*", "", t)
    return t if len(t) <= 48 else t[:46] + "..."


def load(folder):
    state = json.loads((folder / "state.json").read_text())
    recs = [json.loads(x) for x in (folder / "experiments.jsonl").read_text().splitlines() if x.strip()]
    abandoned = sum(r["decision"] == "abandoned" for r in recs)
    recs = [r for r in recs if r["decision"] != "abandoned"]      # lost to a pause; no experiment ran
    promoted = {p["exp"] for p in state.get("promotions", [])}
    agent = state.get("harness") == "agent"
    rows, best, line = [], None, []
    task_toml = Path(state["task_dir"]) / "task.toml"
    maximize = (tomllib.loads(task_toml.read_text()).get("direction", "maximize") == "maximize"
                if task_toml.exists() else True)
    for i, r in enumerate(recs, 1):
        v = r.get("val") or {}
        score = v.get("score")
        kept = r["exp"] in promoted if agent else r["decision"] == "keep"
        if kept and score is not None:
            best = score if best is None or (score > best if maximize else score < best) else best
        rows.append({"finished": i, "exp": r["exp"], "worker": r.get("slot", "-"),
                     "parent": r.get("parent_label") or ("-" if "slot" not in r else ""), "decision": r["decision"],
                     "kept": kept, "val_score": score, "gain": v.get("gain"), "pairs": v.get("pairs"),
                     "title": r.get("title", "")})
        line.append(best)
    return state, rows, line, abandoned


def plot_run(name, state, rows, line, ref_label):
    ref = state["check"]["scores"].get("val") if state["check"]["baseline"] == "reference" else None
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = [r["finished"] for r in rows]
    drop = [r for r in rows if not r["kept"] and r["val_score"] is not None]
    keep = [r for r in rows if r["kept"] and r["val_score"] is not None]
    ax.scatter([r["finished"] for r in drop], [r["val_score"] for r in drop], color="#bbbbbb", s=30,
               label="Discarded", zorder=2)
    ax.scatter([r["finished"] for r in keep], [r["val_score"] for r in keep], color="#2ecc71", edgecolor="#145a32",
               s=70, label="Kept", zorder=3)
    ax.step(xs, [b if b is not None else float("nan") for b in line], where="post", color="#2ecc71", lw=2,
            label="Running best", zorder=1)
    for r in keep:                                    # labels left of late points, right of early ones
        late = r["finished"] > len(rows) / 2
        ax.annotate(short(r["title"]), (r["finished"], r["val_score"]), xytext=(-8 if late else 8, 8),
                    textcoords="offset points", fontsize=8, color="#145a32", ha="right" if late else "left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)
    if ref:
        ax.axhline(ref, color="#c0392b", ls="--", lw=1.2, label=f"{ref_label} (val)")
    unit = "queries/s" if "vector" in name else "tokens/s"
    ax.set_xlabel("Experiments finished")
    ax.set_ylabel(f"Validation score, {unit} (higher is better)")
    ax.set_xticks(xs)
    n_kept = len(keep)
    ax.set_title(f"{RUNS[name][1]} on {RUNS[name][2]}: {len(rows)} experiments, {n_kept} kept")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / f"{name}.png", dpi=120)
    plt.close(fig)


def write_tsv(name, rows):
    with open(HERE / f"{name}.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["finished", "exp", "worker", "parent", "decision", "kept", "val_score", "gain", "pairs", "title"])
        for r in rows:
            w.writerow([r["finished"], r["exp"], r["worker"], r["parent"], r["decision"], "yes" if r["kept"] else "no",
                        "" if r["val_score"] is None else f"{r['val_score']:.6g}",
                        "" if r["gain"] is None else f"{r['gain']:+.3f}", r["pairs"] or "", r["title"]])


def main():
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "runs"
    summary, lines = {}, {}
    for name, (folder, harness, task, ref_label) in RUNS.items():
        state, rows, line, abandoned = load(runs_dir / folder)
        write_tsv(name, rows)
        plot_run(name, state, rows, line, ref_label)
        fin = state.get("final") or {}
        best, ref = fin.get("best", {}), fin.get("reference", {})
        summary[name] = {
            "run": folder, "harness": harness, "task": task, "experiments": len(rows), "abandoned": abandoned,
            "kept": sum(r["kept"] for r in rows), "hours": round(state.get("elapsed_s", 0) / 3600, 2),
            "cost_usd": round(state.get("cost_usd", 0), 2), "best_val": line[-1] if line else None,
            "reference_val": state["check"]["scores"].get("val"),
            "best_test": best.get("median"), "best_test_valid": best.get("valid"),
            "reference_test": ref.get("median"), "reference_test_valid": ref.get("valid"),
            "test_metrics": {k: round(v, 4) for k, v in (best.get("metrics") or {}).items()
                             if isinstance(v, (int, float)) and ("recall" in k or "tokens_per_second" in k)},
            "reference_test_metrics": {k: round(v, 4) for k, v in (ref.get("metrics") or {}).items()
                                       if isinstance(v, (int, float)) and ("recall" in k or "per_second" in k
                                                                           or k.endswith("_qps"))}}
        lines[name] = line
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ("vector-search-v1", "vector-search-v2", "vector-search-v3"):
        line = lines[name]
        xs = range(1, len(line) + 1)
        ax.step(xs, [b if b is not None else float("nan") for b in line], where="post", lw=2.2, color=COLORS[name],
                marker="o", label=f"{RUNS[name][1]}  (test on GIST: {summary[name]['best_test']:,.0f} q/s)")
    ref = summary["vector-search-v1"]["reference_val"]
    ax.axhline(ref, color="#c0392b", ls="--", lw=1.2, label="FAISS, tuned on train (invalid on GIST)")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Experiments finished")
    ax.set_ylabel("Best validation score so far, queries/s")
    ax.set_title("vector-search-multi: three harnesses, 5 experiments each")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "vector-search-comparison.png", dpi=120)
    plt.close(fig)
    print("written:", ", ".join(sorted(p.name for p in HERE.glob("*.tsv")) + sorted(p.name for p in HERE.glob("*.png"))))


if __name__ == "__main__":
    main()
