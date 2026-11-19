"""Interface tests for engine.py. Usage: python test_engine.py <model_dir>

Fast checks only; correctness against the reference is judged by ./dev_eval.
"""
import sys
import unittest

MODEL = sys.argv.pop(1) if len(sys.argv) > 1 else "model"

from engine import Engine  # noqa: E402

ENGINE = Engine(MODEL)
PROMPT = [760, 6511, 314, 9338, 369]          # "The capital of France is"
PARIS = 11751                                 # " Paris"


class EngineTest(unittest.TestCase):
    def test_returns_exactly_n_python_ints(self):
        for n in (1, 2, 7):
            out = ENGINE.generate(list(PROMPT), n)
            self.assertEqual(len(out), n)
            self.assertTrue(all(type(t) is int for t in out))

    def test_greedy_answer(self):
        self.assertEqual(ENGINE.generate(list(PROMPT), 1), [PARIS])

    def test_no_state_between_calls(self):
        first = ENGINE.generate(list(PROMPT), 12)
        ENGINE.generate([9707, 11, 1879, 0] * 20, 5)          # an unrelated request in between
        self.assertEqual(ENGINE.generate(list(PROMPT), 12), first)

    def test_single_token_prompt(self):
        out = ENGINE.generate([760], 4)
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
