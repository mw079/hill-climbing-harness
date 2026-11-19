#!/bin/sh
# Rebuild what llm-inference needs and git does not hold: the Qwen3.5-0.8B weights, the GSM8K
# prompts, and the train/val/test/guard splits with mlx-lm's reference answers; then verify.
#
#   tasks/llm-inference/setup.sh [model] [data] [verify]     (default: all three)
#
# Needs Apple Silicon macOS and `uv`. Downloads about 1.7 GB (model) and a few MB (GSM8K)
# at pinned revisions. FORCE=1 rebuilds steps that are already done.
set -eu
TASK=$(cd "$(dirname "$0")" && pwd)
CACHE=${LLM_CACHE:-$HOME/.cache/hillclimb-llm-inference}
FORCE=${FORCE:-0}
MODEL_REVISION=2fc06364715b967f1860aea9cf38778875588b17
GSM8K_REVISION=740312add88f781978c0658806c59bc2815b9866
STEPS=${*:-model data verify}
REF="uv run -q --no-project --python 3.13 --with mlx==0.32.4 --with mlx-lm==0.32.2 --with pyarrow==21.0.0"
HF="uv run -q --no-project --python 3.13 --with huggingface_hub==0.36.0 hf"
command -v uv >/dev/null || { echo "missing: uv (https://docs.astral.sh/uv/)"; exit 1; }
[ "$(uname -sm)" = "Darwin arm64" ] || { echo "llm-inference needs Apple Silicon macOS (MLX)"; exit 1; }
mkdir -p "$CACHE"

model() {
    if (cd "$TASK" && shasum -a 256 -c --quiet model.sha256 2>/dev/null) && [ "$FORCE" != 1 ]; then
        echo "model: present and verified (FORCE=1 downloads again)"; return
    fi
    $HF download Qwen/Qwen3.5-0.8B --revision "$MODEL_REVISION" --local-dir "$TASK/model"
    (cd "$TASK" && shasum -a 256 -c --quiet model.sha256) || { echo "model: files differ from model.sha256"; exit 1; }
    echo "model: downloaded and verified"
}

data() {
    if [ -f "$TASK/data/train/workload.json" ] && [ "$FORCE" != 1 ]; then
        echo "data: already present (FORCE=1 rebuilds)"; return
    fi
    $HF download openai/gsm8k --repo-type dataset --revision "$GSM8K_REVISION" --include "main/*" \
        --local-dir "$CACHE/gsm8k"
    rm -rf "$CACHE/build"
    mkdir -p "$CACHE/build"
    (cd "$TASK" && $REF python prepare_data.py --reproduce --gsm8k "$CACHE/gsm8k/main" --out "$CACHE/build/data")
    (cd "$TASK" && $REF python prepare_guard.py --gsm8k "$CACHE/gsm8k/main" --out "$CACHE/build/data")
    # Prompts (token ids) must match exactly. The reference answers come from mlx-lm on the GPU
    # and can differ in the last bits on another Apple chip; then this machine's answers are used.
    grep 'workload.json' "$TASK/data.sha256" > "$CACHE/build/prompts.sha256"
    (cd "$CACHE/build" && shasum -a 256 -c --quiet prompts.sha256) || {
        echo "data: the rebuilt prompts differ from the recorded ones; left in $CACHE/build/data"; exit 1; }
    if (cd "$CACHE/build" && shasum -a 256 -c --quiet "$TASK/data.sha256"); then
        echo "data: all splits rebuilt and identical to the recorded ones"
    else
        echo "data: prompts identical; mlx-lm's reference answers differ on this machine and are used."
        echo "      Run hillclimb.py check before comparing results with recorded runs."
    fi
    rm -rf "$TASK/data"
    mv "$CACHE/build/data" "$TASK/data"
}

verify() {
    (cd "$TASK" && shasum -a 256 -c --quiet model.sha256) && echo "verify: model files match"
    if (cd "$TASK" && shasum -a 256 -c --quiet data.sha256 2>/dev/null); then
        echo "verify: all $(wc -l < "$TASK/data.sha256" | tr -d ' ') data files match"
    else
        echo "verify: data differs from the recorded hashes (expected on another chip; see data step)"
    fi
    python3 "$TASK/test_evaluator.py" 2>&1 | tail -1
    echo "Next: python3 hillclimb.py check tasks/llm-inference"
}

for step in $STEPS; do
    case "$step" in model|data|verify) "$step" ;; *) echo "unknown step: $step"; exit 1 ;; esac
done
