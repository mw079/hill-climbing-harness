"""Scoring and correctness rules of the LLM-inference evaluator. No model needed.

    python3 tasks/llm-inference/test_evaluator.py
"""
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("llm_evaluate", Path(__file__).parent / "evaluator/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def truth(tokens, runner_up_gap=2.0):
    """Reference with a runner-up token (id + 1000) `runner_up_gap` below the best."""
    return {"tokens": tokens, "topk_ids": [[t, t + 1000] for t in tokens],
            "topk_logprobs": [[-0.1, -0.1 - runner_up_gap] for _ in tokens]}


class CheckTokens(unittest.TestCase):
    def test_exact_match_checks_everything(self):
        self.assertEqual(ev.check_tokens([1, 2, 3], truth([1, 2, 3])), (3, False, None))

    def test_near_tie_divergence_stops_checking(self):
        checked, diverged, error = ev.check_tokens([1, 1002, 99], truth([1, 2, 3], runner_up_gap=0.25))
        self.assertEqual((checked, diverged, error), (2, True, None))

    def test_clear_wrong_token_is_rejected(self):
        self.assertIsNotNone(ev.check_tokens([1, 1002, 3], truth([1, 2, 3], runner_up_gap=2.0))[2])

    def test_token_outside_top_k_is_rejected(self):
        self.assertIsNotNone(ev.check_tokens([1, 77777, 3], truth([1, 2, 3]))[2])


class Score(unittest.TestCase):
    def setUp(self):
        self.work = {"requests": [{"group": "short", "new_tokens": 4}, {"group": "medium", "new_tokens": 2}]}
        self.truth = {"requests": [truth([1, 2, 3, 4]), truth([5, 6])]}

    def out(self, tokens, seconds=(0.5, 0.25)):
        return {"results": [{"tokens": t, "seconds": s} for t, s in zip(tokens, seconds)],
                "load_seconds": 1.0, "peak_memory_gb": 1.0}

    def test_geometric_mean_of_group_rates(self):
        value, metrics, error = ev.score(self.work, self.truth, self.out([[1, 2, 3, 4], [5, 6]]), elapsed=10)
        self.assertIsNone(error)
        self.assertAlmostEqual(value, math.sqrt(8.0 * 8.0))
        self.assertEqual(metrics["followed_reference_fraction"], 1.0)

    def test_early_divergence_is_valid_when_the_verifier_agrees(self):
        self.truth["requests"][0] = truth([1, 2, 3, 4], runner_up_gap=0.1)
        out = self.out([[1001, 9, 9, 9], [5, 6]])
        value, metrics, error = ev.score(self.work, self.truth, out, elapsed=10, margins=[[0.1, 0, 0, 0], [0, 0]])
        self.assertIsNone(error)
        self.assertEqual(metrics["followed_reference_fraction"], 0.5)   # 1 of 4 plus 2 of 2

    def test_wrong_token_after_divergence_is_rejected_by_the_verifier(self):
        self.truth["requests"][0] = truth([1, 2, 3, 4], runner_up_gap=0.1)
        out = self.out([[1001, 9, 9, 9], [5, 6]])
        _, _, error = ev.score(self.work, self.truth, out, elapsed=10, margins=[[0.1, 0, 7.5, 0], [0, 0]])
        self.assertIn("not a greedy choice", error)

    def test_reported_time_beyond_wall_time_is_rejected(self):
        with self.assertRaises(ValueError):
            ev.score(self.work, self.truth, self.out([[1, 2, 3, 4], [5, 6]], seconds=(5, 5)), elapsed=1)

    def test_memory_limit(self):
        out = self.out([[1, 2, 3, 4], [5, 6]])
        out["peak_memory_gb"] = ev.POLICY["max_peak_memory_gb"] + 1
        self.assertIsNotNone(ev.score(self.work, self.truth, out, elapsed=10)[2])


if __name__ == "__main__":
    unittest.main()
