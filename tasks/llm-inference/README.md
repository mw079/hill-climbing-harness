# Local LLM inference: beat mlx-lm on Qwen3.5-0.8B

The agent improves a plain MLX implementation of Qwen3.5-0.8B (`target/engine.py`, written
for this task: no custom kernels, no `mx.compile`, a Python loop over prompt tokens). The
hidden reference is unmodified `mlx-lm` 0.32.2 (`reference/engine.py` only wraps it).

## Decisions (human, 2026-11-16)

- Model: `Qwen/Qwen3.5-0.8B`, text only. There is no Qwen3.5 0.6B.
- Precision: the official bf16 weights for both sides. No quantization.
- One token per forward pass: no speculative decoding, no MTP head.

## Protocol

- Hardware: Apple M3, 16 GB. MLX 0.32.4 for both; mlx-lm is installed only for the reference.
- Prompts: GSM8K (MIT), tokenized at preparation. Train and val use disjoint halves of GSM8K
  train; test uses GSM8K test.
  - short: one question, about 40-110 tokens, 128 new tokens (4 per split)
  - medium: few-shot, about 500-750 tokens, 64 new tokens (4 per split)
  - long: few-shot, about 2,100-2,200 tokens, 64 new tokens (**test only**)
- Score: geometric mean over groups of new tokens / `generate()` seconds. Load and two
  warm-up requests are untimed. Requests run one at a time.
- Guard: three prompts of about 1,600 tokens (32 new tokens), from GSM8K test rows the test
  split does not use (`prepare_guard.py`). Checked pass/fail only, never scored or shown to
  agents.
- Correctness: while a candidate follows mlx-lm's greedy output, each token must be mlx-lm's
  token or within 0.5 log-probability of its best (a near-tie bf16 rounding can flip). Every
  generated token, including after a divergence, is also re-scored by an evaluator-owned copy
  of the starting engine (`evaluator/verifier_engine.py`) given the candidate's own earlier
  tokens, and must be within 0.5 log-probability of that model's best choice.
- Peak MLX memory at most 6 GB.
- Measurement: 5 repetitions; keep threshold max(3%, 2 x measured noise), 3.0% at the last check (0.3% noise).
- **Why the correctness rule changed (2026-11-17).** The first rule stopped checking a request
  at its first near-tie and required 80% of tokens checked. In the first v3 run
  (`llm-inference-agent-20261117-104848`, archived outside the repository) it rejected all three
  candidates, although every one of their train tokens was within 0.125 log-probability of
  the model's top choice: faster bf16 kernels flip near-ties earlier. Now an evaluator-owned
  copy of the starting engine (`evaluator/verifier_engine.py`) re-scores every generated
  token given the candidate's own prefix (tolerance 0.5). Calibrated on train only:
  reference outputs and all three candidates stay at or below 0.125; the broken engine has
  373 of 768 tokens above 0.5.
- Budget: 10 experiments or 4 hours.

At the last check, on validation: starting engine 32.1 tokens/s, mlx-lm 48.1 tokens/s. Decode
is bandwidth-bound near 65 tokens/s on an M3.

## Set up on a new machine

```sh
tasks/llm-inference/setup.sh                  # model (~1.7 GB), GSM8K, splits, verification
HILLCLIMB_HOME="$PWD/.hillclimb" python3 hillclimb.py check tasks/llm-inference
HILLCLIMB_HOME="$PWD/.hillclimb" python3 hillclimb_agent.py --runs-dir runs run tasks/llm-inference --agent claude-opus-medium
```

`setup.sh` downloads Qwen3.5-0.8B and GSM8K at pinned revisions, checks the weights against
`model.sha256`, and rebuilds every split with `prepare_data.py --reproduce` and
`prepare_guard.py`. Prompts must match `data.sha256` exactly. mlx-lm's reference answers are
computed on the GPU; on another Apple chip they may differ in the last bits, and then that
machine's answers are used (run `check` before comparing with recorded results). Tested on
2026-11-17 from an empty cache: model and all 8 data files matched byte for byte.

## How the task was first built

```sh
REF="uv run --no-project --python 3.13 --with mlx==0.32.4 --with mlx-lm==0.32.2 --with pyarrow==21.0.0"
$REF python tasks/llm-inference/prepare_data.py --gsm8k /path/to/gsm8k/main
$REF python tasks/llm-inference/prepare_guard.py --gsm8k /path/to/gsm8k/main
```

`prepare_data.py` refuses to run on a checked task except with `--reproduce`. To change the
protocol, make a new task copy and keep this one's reports.
