# Task: make local LLM inference faster than MLX's own (mlx-lm)

`engine.py` runs greedy text generation for **Qwen3.5-0.8B** (dense, text-only) on an
Apple M3 (16 GB unified memory, 10-core GPU) with MLX. It is correct but written for clarity, not
speed. Make it faster. The hidden reference is unmodified `mlx-lm` 0.32.2 running the same
model and weights. You see its aggregate validation score only.

## Interface (fixed)

```python
class Engine:
    def __init__(self, model_dir: str): ...
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]: ...
```

`generate` returns exactly `max_new_tokens` greedy token ids as Python ints. It does not
stop at end-of-sequence. Each call is an independent request: no state, cache or result may
carry over between calls (no prompt or prefix caching). Requests arrive one at a time.

## Score and validity

- Score: geometric mean over request groups of generated tokens per second, where a group's
  rate is its total new tokens divided by its total `generate()` time. Prompt processing
  (prefill) is inside the timed call, so both prefill and decode speed count. Model loading
  and two warm-up requests are untimed.
- Train and validation have two groups: **short** prompts (about 40-110 tokens, 128 new
  tokens) and **medium** prompts (about 500-750 tokens, 64 new tokens). The hidden test
  split adds a group of **longer prompts (a few thousand tokens)**. Do not hard-code prompt
  lengths, cache sizes or shapes to the train workload.
- Correctness: greedy decoding within bf16 rounding. While your output follows the
  reference's, each token must be its token or within 0.5 log-probability of its best (a
  near-tie that bf16 rounding can flip). After such a tie the continuations differ, and every
  further token is re-scored by an evaluator-owned copy of the starting engine, given your own
  earlier tokens: it must be within 0.5 log-probability of that model's best choice.
  Changing the order of floating-point operations is fine; changing the model's choices is not.
- Peak MLX memory at most 6 GB.

## Rules

- Weights: use the provided bf16 weights. No quantization, pruning, low-rank
  approximation, or other lossy compression of weights. Activations, KV cache and recurrent
  state may use any precision of 16 bits or more, as long as the correctness check passes.
- One token per forward pass. No speculative decoding, no use of the model's
  multi-token-prediction (`mtp.*`) weights, no draft models.
- Allowed: MLX core (`mlx.core`, `mlx.nn`), `mx.compile`, `mx.fast.*`, and custom Metal
  kernels via `mx.fast.metal_kernel`. Python standard library.
- Not allowed: mlx-lm (it is not installed for you), other inference engines or libraries,
  or source copied from them. Write your own implementation.
- Do not read files other than the model directory, cache answers, inspect the driver, or
  interfere with timing.
- Read the model configuration from `config.json`; do not assume this exact model's sizes
  where a value is available from the configuration.

## Model facts (from config.json)

24 layers: 18 Gated DeltaNet linear-attention layers and every 4th layer full attention
(8 query heads, 2 KV heads, head dim 256, output gate, RoPE on the first 64 dims, theta
1e7). Hidden size 1024, MLP 3584 (SwiGLU), vocabulary 248,320 with tied embeddings. The
linear-attention layers have 16 heads of key/value dim 128, a depthwise causal conv of
width 4, and a float32 recurrent state per head. Weights in bf16, about 1.5 GB for the text
model; at the M3's ~100 GB/s memory bandwidth, reading every weight once per token caps
decode near 65 tokens/s.

## Files and development

- `engine.py`: implement here. Helpers under `src/` are allowed, imported by engine.py.
- `test_engine.py`: interface tests, run as `python3 test_engine.py <model_dir>` by the
  controller. Preserve these tests; add your own.
- `./dev_eval`: runs the official evaluator on the train split and prints aggregates.
- `data/train/workload.json`: the train prompts (token ids), and `data/train/truth.json`:
  the reference's greedy tokens and top-8 log-probabilities for them.
- The model directory is `{task_dir}/model`.
  Use it for your own tests and timing. Read only that folder: the rest of the task folder
  (evaluator, reference, hidden splits) is off limits.
