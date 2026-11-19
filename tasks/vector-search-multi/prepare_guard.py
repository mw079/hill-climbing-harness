"""Build the guard split: synthetic clustered vectors of a dimension no other split uses.

Checked pass/fail only (recall floor), never scored, never shown to agents. It catches engines
whose constants only fit SIFT and Deep. Layout: data/guard/synth/{base.bin, queries.bin,
exact_ids.npy, kth.npy, metadata.json}.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prep", TASK / "prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
SEED, DIM, CLUSTERS, BASE, QUERIES = 20261117, 200, 48, 12000, 512


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=TASK / "data", help="where the data folder is")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    centres = rng.normal(0, 4.0, (CLUSTERS, DIM))
    scales = rng.uniform(0.3, 2.5, (CLUSTERS, DIM))       # anisotropic, uneven clusters
    weights = rng.dirichlet(np.ones(CLUSTERS) * 0.7)

    def sample(n):
        c = rng.choice(CLUSTERS, n, p=weights)
        return (centres[c] + rng.normal(0, 1, (n, DIM)) * scales[c]).astype(np.float32)

    base, queries = sample(BASE), sample(QUERIES)
    out = args.out / "guard" / "synth"
    out.mkdir(parents=True, exist_ok=True)
    ids, kth = prep.ground_truth(base, queries)
    prep.write_matrix(out / "base.bin", base)
    prep.write_matrix(out / "queries.bin", queries)
    np.save(out / "exact_ids.npy", ids, allow_pickle=False)
    np.save(out / "kth.npy", kth, allow_pickle=False)
    (out / "metadata.json").write_text(json.dumps({"split": "guard", "dataset": "synthetic clustered Gaussian",
        "base_vectors": BASE, "queries": QUERIES, "dimension": DIM, "seed": SEED}, indent=2) + "\n")
    print("guard", base.shape, queries.shape)


if __name__ == "__main__":
    main()
