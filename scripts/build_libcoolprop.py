#!/usr/bin/env python
"""Build CoolProp's shared library from source and copy it into the package, so
maturin bundles it into the wheel (the plugin dlopens it at runtime).

Cross-platform (Linux / macOS / Windows). Run before building the wheel -- this is
the cibuildwheel ``before-all`` step. Requires git, cmake and a C++ compiler.
CoolProp ships no prebuilt 8.0 shared libraries, hence the from-source build.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

COOLPROP_VERSION = "v8.0.0"  # must match the pinned Python `coolprop`
PROJECT = Path(__file__).resolve().parent.parent  # repo root
PKG = PROJECT / "encomp" / "coolprop"


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    work = Path(os.environ.get("COOLPROP_BUILD_DIR", PROJECT / "_coolprop_build"))
    src = work / "CoolProp"
    if not (src / "CMakeLists.txt").exists():
        work.mkdir(parents=True, exist_ok=True)
        run(
            "git", "clone", "--depth", "1", "--branch", COOLPROP_VERSION,
            "--recursive", "--shallow-submodules",
            "https://github.com/CoolProp/CoolProp.git", str(src),
        )  # fmt: skip

    # CoolProp hardcodes `-m${BITNESS}` (= -m64), which gcc rejects on aarch64. All
    # our targets are 64-bit (the compiler default), so strip the flag for portability.
    cml = src / "CMakeLists.txt"
    cml.write_text(cml.read_text().replace("-m${BITNESS}", ""))

    build = src / "build"
    build.mkdir(exist_ok=True)
    cmake_args = ["cmake", "..", "-DCOOLPROP_SHARED_LIBRARY=ON", "-DCMAKE_BUILD_TYPE=Release"]
    # match the wheel's target arch(es); cibuildwheel sets ARCHFLAGS on macOS
    archflags = os.environ.get("ARCHFLAGS", "")
    if sys.platform == "darwin":
        arches = [a for a in archflags.split() if a not in ("-arch", "")]
        if arches:
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=" + ";".join(arches))
    run(*cmake_args, cwd=build)
    run("cmake", "--build", ".", "--config", "Release", "-j", "4", cwd=build)

    candidates = ["libCoolProp.dylib", "libCoolProp.so", "CoolProp.dll", "libCoolProp.dll"]
    found = next((p for name in candidates for p in build.rglob(name)), None)
    if found is None:
        sys.exit(f"libCoolProp not found under {build}")
    PKG.mkdir(parents=True, exist_ok=True)
    for stale in candidates:  # drop any other-platform lib so the wheel ships only this one
        (PKG / stale).unlink(missing_ok=True)
    dest = PKG / found.name
    shutil.copy2(found, dest)
    print(f"bundled {found} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
