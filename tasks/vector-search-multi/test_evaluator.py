"""Multi-dataset scoring rules for tasks/vector-search-multi. Uses pinned NumPy."""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("multi_evaluate", TASK / "evaluator/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
prep_spec = importlib.util.spec_from_file_location("multi_prepare", TASK / "prepare_data.py")
prep = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prep)

# Returns IDs 0..9: correct when the base has exactly 10 rows, wrong when the nearest
# rows are elsewhere. Output plumbing only, not a search algorithm.
HEADER = '''#pragma once
#include <cstdint>
class Engine { public:
 Engine(const float*, uint32_t, uint32_t) {}
 void search(const float*, uint32_t k, uint32_t* ids) { for (uint32_t i=0;i<k;++i) ids[i]=i; }
};
'''


class MultiDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        (self.candidate / "engine.h").write_text(HEADER)

    def dataset(self, split, name, base, queries):
        d = self.root / split / name
        d.mkdir(parents=True)
        prep.write_matrix(d / "base.bin", base)
        prep.write_matrix(d / "queries.bin", queries)
        np.save(d / "kth.npy", prep.ground_truth(base, queries)[1])

    def test_all_datasets_valid_scores_geometric_mean(self):
        a = np.arange(40, dtype=np.float32).reshape(10, 4)
        b = np.arange(80, dtype=np.float32).reshape(10, 8)
        self.dataset("ok", "a", a, a)
        self.dataset("ok", "b", b, b)
        r = ev.evaluate(self.candidate, self.root / "ok")
        self.assertTrue(r["valid"], r)
        m = r["metrics"]
        self.assertAlmostEqual(r["score"], math.sqrt(m["a_qps"] * m["b_qps"]), delta=1e-6 * r["score"])
        self.assertEqual(m["min_recall_at_10"], 1.0)

    def test_one_dataset_below_floor_invalidates(self):
        good = np.arange(40, dtype=np.float32).reshape(10, 4)
        # 20 rows; queries sit on rows 10..19, so IDs 0..9 are wrong.
        bad = np.arange(80, dtype=np.float32).reshape(20, 4) * 10
        self.dataset("mixed", "good", good, good)
        self.dataset("mixed", "bad", bad, bad[10:])
        r = ev.evaluate(self.candidate, self.root / "mixed")
        self.assertFalse(r["valid"])
        self.assertIn("1 of 2", r["error"])
        self.assertLess(r["metrics"]["min_recall_at_10"], 0.95)

    def test_single_dataset_layout_still_works(self):
        a = np.arange(40, dtype=np.float32).reshape(10, 4)
        d = self.root / "flat"
        d.mkdir()
        prep.write_matrix(d / "base.bin", a)
        prep.write_matrix(d / "queries.bin", a)
        np.save(d / "kth.npy", prep.ground_truth(a, a)[1])
        self.assertTrue(ev.evaluate(self.candidate, d)["valid"])

    def test_prepared_splits_are_disjoint_kinds(self):
        kinds = {s: sorted(p.name for p in (TASK / "data" / s).iterdir()) for s in ("train", "val", "test")}
        self.assertEqual(kinds["train"], kinds["val"])
        self.assertFalse(set(kinds["test"]) & set(kinds["train"]))


if __name__ == "__main__":
    unittest.main()
