"""Prepare multi-dataset splits from raw public vectors; imports no search-engine code.

Train and val each hold SIFT and Deep subsets (disjoint rows between train and val).
Test holds GIST only: a vector kind the loop never measures, to check generalisation.
Layout: data/<split>/<dataset>/{base.bin, queries.bin, exact_ids.npy, kth.npy, metadata.json}.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

TASK = Path(__file__).resolve().parent
SEED = 20261117


def fvecs(path):
    raw = np.memmap(path, dtype="<i4", mode="r")
    d = int(raw[0])
    rows = raw.reshape(-1, d + 1)
    if not np.all(rows[:1000, 0] == d) or not np.all(rows[-1000:, 0] == d):
        raise ValueError("inconsistent fvecs dimensions")
    return rows[:, 1:].view("<f4")


def write_matrix(path, values):
    values = np.ascontiguousarray(values, dtype="<f4")
    with open(path, "wb") as f:
        np.array(values.shape, dtype="<u4").tofile(f)
        values.tofile(f)


def ground_truth(base, queries):
    b = base.astype(np.float64)
    bnorm = np.einsum("nd,nd->n", b, b)
    ids, kth = [], []
    for start in range(0, len(queries), 32):
        q = queries[start:start + 32].astype(np.float64)
        distances = np.einsum("qd,qd->q", q, q)[:, None] + bnorm - 2 * q @ b.T
        np.maximum(distances, 0, out=distances)
        selected = np.argpartition(distances, 9, axis=1)[:, :10]
        ids.append(selected.astype("<u4"))
        kth.append(np.max(np.take_along_axis(distances, selected, axis=1), axis=1))
    return np.concatenate(ids), np.concatenate(kth)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sift", type=Path, required=True, help="directory with base.fvecs/query.fvecs")
    p.add_argument("--deep", type=Path, required=True, help="directory with base.fvecs/query.fvecs")
    p.add_argument("--gist", type=Path, required=True, help="directory with base.fvecs/query.fvecs")
    p.add_argument("--base-count", type=int, default=12000)
    p.add_argument("--query-count", type=int, default=512)
    p.add_argument("--out", type=Path, default=TASK / "data", help="where to write the splits")
    p.add_argument("--reproduce", action="store_true",
                   help="rebuild the recorded splits (setup.sh); allowed on a checked task, keeps preparation.json")
    args = p.parse_args()
    if (TASK / "check_report.json").exists() and not args.reproduce:
        p.error("task is already checked; use a new task copy for a new data protocol")
    rng = np.random.default_rng(SEED)
    nb, nq = args.base_count, args.query_count
    sources = {"sift": (args.sift, "SIFT", False), "deep": (args.deep, "Deep normalized to unit length", True),
               "gist": (args.gist, "GIST", False)}
    raw = {k: (fvecs(d / "base.fvecs"), fvecs(d / "query.fvecs")) for k, (d, _, _) in sources.items()}
    # Two disjoint selections (train, val) from SIFT and Deep; one (test) from GIST.
    picks = {}
    for key, copies in (("sift", 2), ("deep", 2), ("gist", 1)):
        xb, xq = raw[key]
        if len(xb) < copies * nb or len(xq) < copies * nq:
            p.error(f"not enough vectors in {key}")
        bi, qi = rng.permutation(len(xb))[:copies * nb], rng.permutation(len(xq))[:copies * nq]
        picks[key] = [(bi[i * nb:(i + 1) * nb], qi[i * nq:(i + 1) * nq]) for i in range(copies)]
    plan = {"train": [("sift", 0), ("deep", 0)], "val": [("sift", 1), ("deep", 1)], "test": [("gist", 0)]}
    provenance = {"seed": SEED, "base_count": nb, "query_count": nq,
                  "sources": {k: str(d.resolve()) for k, (d, _, _) in sources.items()},
                  "plan": {s: [k for k, _ in v] for s, v in plan.items()}, "split_indices": {}}
    for split, members in plan.items():
        for key, copy in members:
            out = args.out / split / key
            out.mkdir(parents=True, exist_ok=True)
            xb, xq = raw[key]
            bi, qi = picks[key][copy]
            base, queries = np.array(xb[np.sort(bi)]), np.array(xq[np.sort(qi)])
            if not np.isfinite(base).all() or not np.isfinite(queries).all():
                raise ValueError("non-finite source values")
            if sources[key][2]:
                base /= np.linalg.norm(base, axis=1, keepdims=True)
                queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            ids, kth = ground_truth(base, queries)
            write_matrix(out / "base.bin", base)
            write_matrix(out / "queries.bin", queries)
            np.save(out / "exact_ids.npy", ids, allow_pickle=False)
            np.save(out / "kth.npy", kth, allow_pickle=False)
            (out / "metadata.json").write_text(json.dumps({"split": split, "dataset": sources[key][1],
                "base_vectors": len(base), "queries": len(queries), "dimension": base.shape[1]}, indent=2) + "\n")
            provenance["split_indices"][f"{split}/{key}"] = {"base": np.sort(bi).tolist(),
                "queries": np.sort(qi).tolist(),
                "sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()}}
            print(split, key, base.shape, queries.shape, flush=True)
    for key in ("sift", "deep"):
        (tb, tq), (vb, vq) = picks[key]
        assert not set(tb) & set(vb) and not set(tq) & set(vq)
    if not args.reproduce:
        (TASK / "preparation.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
