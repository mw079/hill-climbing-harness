"""Import only FAISS public headers, static library, license, and build provenance."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

TASK = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="FAISS source directory")
    p.add_argument("--build", type=Path, required=True, help="FAISS CMake build directory")
    p.add_argument("--reproduce", action="store_true",
                   help="install a rebuilt library on a checked task (setup.sh); the frozen setting is unchanged")
    args = p.parse_args()
    if (TASK / "check_report.json").exists() and not args.reproduce:
        p.error("reference is frozen; use a new task copy to replace it")
    vendor = TASK / "reference/vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    for header in (args.source / "faiss").rglob("*.h"):
        dst = vendor / "include/faiss" / header.relative_to(args.source / "faiss")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(header, dst)
    shutil.copyfile(args.build / "faiss/libfaiss.a", vendor / "libfaiss.a")
    shutil.copyfile(args.source / "LICENSE", vendor / "LICENSE")
    flags = (args.build / "faiss/CMakeFiles/faiss.dir/flags.make").read_text()
    (vendor / "BUILD_INFO.json").write_text(json.dumps({
        "origin": "FAISS source build; no custom vector engine code",
        "library_sha256": hashlib.sha256((vendor / "libfaiss.a").read_bytes()).hexdigest(),
        "compiler_flags": next(x for x in flags.splitlines() if x.startswith("CXX_FLAGS =")),
        "platform": "Apple Silicon macOS",
        "headers": len(list((vendor / "include").rglob("*.h")))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
