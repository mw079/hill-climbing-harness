#!/bin/sh
# Rebuild what vector-search-multi needs and git does not hold: the FAISS reference library,
# the source datasets, and the train/val/test/guard splits; then verify every data file.
#
#   tasks/vector-search-multi/setup.sh [faiss] [data] [verify]     (default: all three)
#
# Needs Apple Silicon macOS with Xcode command line tools, Homebrew `cmake` and `libomp`,
# `uv`, `git` and `curl`. Downloads about 7.9 GB into $VS_CACHE (default
# ~/.cache/hillclimb-vector-search). FORCE=1 rebuilds steps that are already done.
set -eu
TASK=$(cd "$(dirname "$0")" && pwd)
CACHE=${VS_CACHE:-$HOME/.cache/hillclimb-vector-search}
FORCE=${FORCE:-0}
FAISS_TAG=v1.12.0
FAISS_COMMIT=e8234e563f1ecef5f036e83c3cfee366d3f1fbca
STEPS=${*:-faiss data verify}
py() { uv run -q --no-project --python 3.13 --with numpy==2.2.6 "$@"; }
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)"; exit 1; }; }

need uv "https://docs.astral.sh/uv/"
need git "xcode-select --install"
need curl "xcode-select --install"
mkdir -p "$CACHE"

faiss() {
    if [ -f "$TASK/reference/vendor/libfaiss.a" ] && [ "$FORCE" != 1 ]; then
        echo "faiss: reference/vendor is installed (FORCE=1 rebuilds)"; return
    fi
    need cmake "brew install cmake"
    need brew "https://brew.sh"
    OMP=$(brew --prefix libomp 2>/dev/null) || true      # prints a path even when not installed
    [ -f "$OMP/lib/libomp.dylib" ] || { echo "missing: libomp (brew install libomp)"; exit 1; }
    SRC="$CACHE/faiss"
    [ -d "$SRC/.git" ] || git clone --quiet --depth 1 --branch "$FAISS_TAG" \
        https://github.com/facebookresearch/faiss.git "$SRC"
    [ "$(git -C "$SRC" rev-parse HEAD)" = "$FAISS_COMMIT" ] || { echo "faiss: unexpected commit"; exit 1; }
    # The options of the recorded build (reference/vendor/BUILD_INFO.json): Release, -O3
    # -mcpu=native, Apple Accelerate BLAS, Homebrew OpenMP, no GPU.
    cmake -S "$SRC" -B "$SRC/build" -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -mcpu=native" \
        -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF -DBUILD_TESTING=OFF \
        -DBLA_VENDOR=Apple \
        -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_CXX_LIB_NAMES=omp \
        -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_C_LIB_NAMES=omp \
        -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib" >/dev/null
    cmake --build "$SRC/build" --target faiss -j "$(sysctl -n hw.ncpu)"
    py python "$TASK/prepare_reference.py" --reproduce --source "$SRC" --build "$SRC/build"
    echo "faiss: installed into reference/vendor"
    echo "  The frozen FAISS setting (reference/frozen.h) is unchanged. The library is compiled for"
    echo "  this machine (-mcpu=native), so its hash differs from the recorded one on other chips."
}

download() {   # url, file, expected size in bytes
    f="$CACHE/downloads/$2"
    mkdir -p "$CACHE/downloads"
    if [ ! -f "$f" ] || [ "$(stat -L -f %z "$f")" != "$3" ]; then
        echo "downloading $2"
        curl -fL --retry 3 -C - -o "$f" "$1"
    fi
    [ "$(stat -L -f %z "$f")" = "$3" ] || { echo "$2: unexpected size"; exit 1; }
}

data() {
    if [ -f "$TASK/data/train/sift/base.bin" ] && [ "$FORCE" != 1 ]; then
        echo "data: already present (FORCE=1 rebuilds)"; return
    fi
    download ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz sift.tar.gz 168280445
    download http://ann-benchmarks.com/deep-image-96-angular.hdf5 deep-image-96-angular.hdf5 3848008288
    download http://ann-benchmarks.com/gist-960-euclidean.hdf5 gist-960-euclidean.hdf5 3844648288
    py --with h5py==3.14.0 python "$TASK/convert_sources.py" "$CACHE/downloads" "$CACHE/sources"
    rm -rf "$CACHE/build"
    mkdir -p "$CACHE/build"
    py python "$TASK/prepare_data.py" --reproduce --out "$CACHE/build/data" \
        --sift "$CACHE/sources/sift1m" --deep "$CACHE/sources/deep1m" --gist "$CACHE/sources/gist1m"
    py python "$TASK/prepare_guard.py" --out "$CACHE/build/data"
    (cd "$CACHE/build" && shasum -a 256 -c --quiet "$TASK/data.sha256") || {
        echo "data: the rebuilt splits differ from the recorded ones; left in $CACHE/build/data"; exit 1; }
    rm -rf "$TASK/data"
    mv "$CACHE/build/data" "$TASK/data"
    echo "data: all splits rebuilt and identical to the recorded ones"
}

verify() {
    (cd "$TASK" && shasum -a 256 -c --quiet data.sha256) && echo "verify: all $(wc -l < "$TASK/data.sha256" | tr -d ' ') data files match"
    [ -f "$TASK/reference/vendor/libfaiss.a" ] || { echo "verify: reference/vendor/libfaiss.a is missing"; exit 1; }
    py python "$TASK/test_evaluator.py" 2>&1 | tail -1
    echo "Next: python3 hillclimb.py check tasks/vector-search-multi"
}

for step in $STEPS; do
    case "$step" in faiss|data|verify) "$step" ;; *) echo "unknown step: $step"; exit 1 ;; esac
done
