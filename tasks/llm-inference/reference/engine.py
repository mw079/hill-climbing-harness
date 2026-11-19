"""Hidden reference: unmodified mlx-lm greedy generation behind the task's Engine interface.

The candidate environment does not have mlx-lm installed; only the reference does.
"""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step


def greedy(logits):
    return mx.argmax(logits, axis=-1)


class Engine:
    def __init__(self, model_dir):
        self.model, self.tokenizer = load(str(model_dir))

    def steps(self, prompt_ids, max_new_tokens):
        """mlx-lm's own generation loop: yields (token, logprobs) per new token."""
        yield from generate_step(mx.array(prompt_ids), self.model, max_tokens=max_new_tokens,
                                 sampler=greedy)

    def generate(self, prompt_ids, max_new_tokens):
        return [int(token) for token, _ in self.steps(prompt_ids, max_new_tokens)]
