# Vector search, multi-dataset (goal: a general engine)

Build a vector-search engine in C++ from an empty interface and beat a hidden, train-tuned
FAISS reference, on data the loop never measured. It replaced a first version (SIFT for
train and validation, Deep for test) in which an engine that beat FAISS on validation missed
the recall floor on test because its search effort was hard-coded.

## Protocol

- Train and val: SIFT (128-d) and Deep (96-d, normalized), 12,000 base vectors and 512
  queries each, with disjoint rows between train and val.
- Test: GIST (960-d), a kind of vector the loop never measures.
- Score: geometric mean of per-dataset QPS. Valid only if recall@10 >= 0.95 on **every**
  dataset in the split.
- Guard: synthetic clustered vectors with 200 dimensions (`prepare_guard.py`), checked
  pass/fail only (recall floor), never scored or shown to agents.
- Timing: at least 1.0 s per measurement and 5 repetitions. The last check measured 1.7%
  noise, so the keep threshold is 3.4% (max of 3% and 2 x noise).
- Budget: 10 experiments or 3 hours.
- FAISS: a catalog of 13 settings (Flat; HNSW16 and HNSW32 at efSearch 32/64/128/256;
  IVF64,Flat at nprobe 8/16/32/64), timed on train only (`tune_reference.py`). The fastest with
  recall >= 0.95 on every train dataset is frozen: HNSW16, efSearch 32.

## Known result before any run

The frozen FAISS setting reaches recall 0.907 on the GIST test, below the floor. The
human decision (2026-11-16) was to keep it as tuned rather than retune after seeing
test. The harness reports this as a warning, and the final report states it.

## Set up on a new machine

```sh
tasks/vector-search-multi/setup.sh            # FAISS v1.12.0, datasets (~7.9 GB), splits, verification
HILLCLIMB_HOME="$PWD/.hillclimb" python3 hillclimb.py check tasks/vector-search-multi
HILLCLIMB_HOME="$PWD/.hillclimb" python3 hillclimb_agent.py --runs-dir runs run tasks/vector-search-multi --agent claude-opus-medium
```

`setup.sh` builds FAISS v1.12.0 (commit `e8234e5`) with the recorded options and installs it
in `reference/vendor`; the frozen FAISS setting (`reference/frozen.h`) is in git. It downloads
SIFT1M (INRIA TEXMEX) and the Deep and GIST files of ann-benchmarks, converts them as the task
was built (`convert_sources.py`: Deep's first million base vectors, unit-normalized), rebuilds
every split with `prepare_data.py --reproduce` and `prepare_guard.py`, and installs them only if
all 30 files match `data.sha256` byte for byte. Tested on 2026-11-17: every file matched. The
FAISS library is compiled for the local chip, so its hash differs from the recorded build;
headers and compiler flags match. Steps can be run alone: `setup.sh faiss`, `data`, `verify`.

## How the task was first built

```sh
D=/path/to/source/fvecs
PY="uv run --no-project --python 3.13 --with numpy==2.2.6 python"
$PY tasks/vector-search-multi/prepare_data.py --sift $D/sift1m --deep $D/deep1m --gist $D/gist1m
$PY tasks/vector-search-multi/prepare_guard.py
$PY tasks/vector-search-multi/tune_reference.py
```
