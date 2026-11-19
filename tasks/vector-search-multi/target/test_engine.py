"""Small interface/correctness oracle, not a starting search implementation."""
import json
import math
import random
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


def write_matrix(path, rows):
    flat = [float(x) for row in rows for x in row]
    path.write_bytes(struct.pack("<II", len(rows), len(rows[0])) + struct.pack("<" + "f" * len(flat), *flat))


class EngineTest(unittest.TestCase):
    def check_case(self, base, queries):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_matrix(tmp / "base.bin", base)
            write_matrix(tmp / "queries.bin", queries)
            result = subprocess.run([str(Path("engine").resolve()), str(tmp / "base.bin"),
                                     str(tmp / "queries.bin"), str(tmp / "ids.bin"),
                                     str(tmp / "metrics.json"), "0.02", "20", "512"],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            raw = (tmp / "ids.bin").read_bytes()
            self.assertEqual(len(raw), len(queries) * 10 * 4)
            ids = struct.unpack("<" + "I" * (len(queries) * 10), raw)
            correct = 0
            for i, q in enumerate(queries):
                row = ids[i * 10:(i + 1) * 10]
                self.assertEqual(len(set(row)), 10)
                self.assertTrue(all(x < len(base) for x in row))
                distances = [sum((a - b) ** 2 for a, b in zip(q, x)) for x in base]
                cutoff = sorted(distances)[9]
                correct += sum(distances[x] <= cutoff + 1e-5 * max(1, cutoff) for x in row)
            self.assertGreaterEqual(correct / (len(queries) * 10), 0.95)
            metrics = json.loads((tmp / "metrics.json").read_text())
            self.assertTrue(math.isfinite(metrics["search_seconds"]))

    def test_random_small_index(self):
        rng = random.Random(41)
        self.check_case([[rng.uniform(-2, 2) for _ in range(8)] for _ in range(32)],
                        [[rng.uniform(-2, 2) for _ in range(8)] for _ in range(16)])

    def test_ties_and_duplicate_vectors_have_distinct_ids(self):
        self.check_case([[float(i // 4)] * 8 for i in range(32)], [[0.5] * 8 for _ in range(12)])

    def test_identical_base_and_query_vectors(self):
        base = [[float(i), 0.0, -float(i), 1.0] for i in range(24)]
        self.check_case(base, base[:12])


if __name__ == "__main__":
    unittest.main()
