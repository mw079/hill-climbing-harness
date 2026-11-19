# Task: build a vector-search engine from scratch and beat a hidden FAISS reference

Implement `Engine` in `engine.h`. The supplied file is an unimplemented interface stub.
No existing custom search engine or reference implementation is in this workspace.
First create a valid engine, then improve it. A first valid candidate is kept even if
it is slower than FAISS; subsequent keeps require beating the current best.

## The goal is a general engine

Every split contains **several datasets of different kinds** (different dimensions, value
ranges, and structure). Train has two, in `data/train/<name>/`. The hidden validation
split has the same kinds with different vectors. The final hidden test uses **a kind of
vector data that appears in neither train nor validation**. An engine whose constants
only fit the training datasets (fixed probe counts, value-range or integer shortcuts,
dimension-specific code paths without a general fallback) is likely to fail there.
Prefer settings derived from the data at build time over hard-coded numbers.

## Objective and fixed constraints

- Score: geometric mean over the split's datasets of queries per second, higher is better.
  Each dataset counts equally, whatever its dimension. One query per `search` call,
  one CPU thread. Each dataset runs in its own process with a fresh `Engine`.
- Distance: squared Euclidean distance over finite float32 vectors, `k = 10`.
- Validity: recall@10 at least **0.95 on every dataset** in the split, distinct in-range
  IDs, deterministic results. One dataset below the floor makes the candidate invalid.
- Build time at most **20 seconds**; peak resident memory at most **512 MiB** for the
  process, including input arrays and index. The process has a 45-second wall timeout.
- C++20 on Apple Silicon macOS, `/usr/bin/clang++ -O3 -mcpu=native -DNDEBUG`.
  The standard library, compiler SIMD intrinsics, and Apple Accelerate BLAS/LAPACK are
  allowed. Other packages, ANN libraries, FAISS, and copied search-engine source are not.
- Reading papers/algorithm descriptions is allowed when web access is enabled. This
  pilot has web research disabled. Derive and write your own implementation.

FAISS is independently timed by the controller under the same query mode, thread count,
compiler optimization level, recall floor, and resource caps. Its settings are selected
on train (one setting for all datasets) and frozen. The reference is a finite tested FAISS catalog, not a claim about
the best possible FAISS configuration. You see its aggregate validation score only.

## Interface

```cpp
class Engine {
public:
    Engine(const float* base, uint32_t n, uint32_t dimension);
    void search(const float* query, uint32_t k, uint32_t* ids);
};
```

The base is row-major, with `n` vectors of `dimension` float32 values. Its storage remains
valid for the index lifetime. Do not mutate base/query data. `search` writes exactly `k`
distinct IDs in `[0, n)`. Return order need not be sorted. Repeated searches of the same
query in one index must return the same IDs. Support varying `n >= 10` and dimensions,
not just the training sizes. Queries can equal stored vectors, and vectors can tie.

The driver owns all file I/O, input loading, timers, warmup, repeatability checks, and
resource accounting. Implement only the index and search; do not open files, access
the environment, inspect arguments, manipulate clocks, create threads/processes, or
cache query answers. Index/query scratch buffers may be reused; outputs must be computed
from each query. Do not alter or interfere with the driver or checks using macros,
interposition, undefined behavior, or benchmark-specific branches.

## Files and development

- `engine.h`: implement here; header helpers under `src/` are allowed and must be included
  by engine.h. The official compiler compiles the fixed driver as one translation unit.
- `driver.cpp`, `Makefile`: supplied interface and timing infrastructure, not editable.
- `test_engine.py`: tiny correctness oracles. Preserve these tests; add edge cases.
- `make` builds `./engine`. `python3 -m unittest -q test_engine` runs interface tests.
- `./dev_eval` recompiles with the controller's driver and scores only train. Timed
  work includes search plus the same small output-repeatability check for both engines;
  compilation, index construction, file I/O, and warmup are outside the QPS score.
- `data/train/<name>/base.bin` and `queries.bin` (one folder per dataset): little-endian uint32 row count, uint32 dimension,
  then contiguous little-endian float32 values. Training exact IDs and kth-neighbor
  distances are available as `.npy` files, and shapes are in `metadata.json`.

The scorer independently computes distances in float64 and accepts distinct IDs whose
distance is within `1e-6 * max(1, kth_distance)` of the exact tenth neighbor. It checks all
queries; shape errors, duplicate/out-of-range IDs, non-finite outputs, and insufficient
recall invalidate the candidate. Dataset-shape tuning and parameter selection must use
train only. Never inspect hidden splits or choose parameters using their scores.

The evaluator rebuilds the candidate in a fresh temporary directory and never uses an
agent-built binary for official scoring. Hidden truth and original split paths are not
passed to the candidate process. Do not read the hidden splits or the evaluator: depending on
the harness, this is enforced by permission rules and a sandbox, or only by these instructions.
