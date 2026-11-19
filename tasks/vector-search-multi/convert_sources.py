"""Convert the downloaded source datasets to the fvecs files prepare_data.py reads.

    python convert_sources.py <downloads> <sources>

<downloads> holds sift.tar.gz (INRIA TEXMEX), deep-image-96-angular.hdf5 and
gist-960-euclidean.hdf5 (ann-benchmarks). Writes <sources>/{sift1m,deep1m,gist1m}/{base,query}.fvecs
exactly as the files the task was prepared from: SIFT as published; Deep as the first million
base vectors and all queries, each normalized to unit length in float32; GIST as published.
"""
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np


def write_fvecs(path, values, chunk=100_000):
    """Stream rows to fvecs (int32 dimension, then float32 values) without loading everything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n, d = values.shape
    with open(path, "wb") as f:
        for start in range(0, n, chunk):
            block = np.asarray(values[start:start + chunk], dtype=np.float32)
            out = np.empty((len(block), d + 1), dtype=np.float32)
            out[:, 0] = np.array(d, dtype=np.int32).view(np.float32)
            out[:, 1:] = block
            f.write(out.tobytes())


def normalise(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    downloads, sources = Path(sys.argv[1]), Path(sys.argv[2])
    sift = sources / "sift1m"
    if not (sift / "query.fvecs").exists():
        with tarfile.open(downloads / "sift.tar.gz") as tar:
            for member, name in (("sift/sift_base.fvecs", "base.fvecs"), ("sift/sift_query.fvecs", "query.fvecs")):
                sift.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(sift / name, "wb") as dst:
                    while chunk := src.read(1 << 24):
                        dst.write(chunk)
        print("sift1m written")
    if not (sources / "deep1m" / "query.fvecs").exists():
        with h5py.File(downloads / "deep-image-96-angular.hdf5", "r") as f:
            write_fvecs(sources / "deep1m" / "base.fvecs", normalise(f["train"][:1_000_000]))
            write_fvecs(sources / "deep1m" / "query.fvecs", normalise(f["test"][:]))
        print("deep1m written")
    if not (sources / "gist1m" / "query.fvecs").exists():
        with h5py.File(downloads / "gist-960-euclidean.hdf5", "r") as f:
            write_fvecs(sources / "gist1m" / "base.fvecs", f["train"])
            write_fvecs(sources / "gist1m" / "query.fvecs", f["test"][:])
        print("gist1m written")


if __name__ == "__main__":
    main()
