#!/usr/bin/env python
"""Build CoolProp's shared library from source and copy it into the package, so
maturin bundles it into the wheel (the plugin dlopens it at runtime).

Cross-platform (Linux / macOS / Windows). Run before building the wheel -- this is
the cibuildwheel ``before-all`` step. Requires git, cmake and a C++ compiler.
CoolProp ships no prebuilt 8.0 shared libraries, hence the from-source build.
"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # repo root
PKG = PROJECT / "encomp" / "coolprop"
BUILD_INFO = runpy.run_path(str(PKG / "_build_info.py"))
COOLPROP_VERSION = str(BUILD_INFO["BUNDLED_COOLPROP_TAG"])
COOLPROP_COMMIT = str(BUILD_INFO["BUNDLED_COOLPROP_COMMIT"])


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def verify_commit(src: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=src,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if head != COOLPROP_COMMIT:
        raise RuntimeError(
            f"CoolProp {COOLPROP_VERSION} resolved to commit {head}, expected {COOLPROP_COMMIT}. "
            "The upstream tag moved; do not build from it until this is understood."
        )

    print(f"+ verified CoolProp {COOLPROP_VERSION} at {head}", flush=True)


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

    verify_commit(src)

    # CoolProp hardcodes `-m${BITNESS}` (= -m64), which gcc rejects on aarch64. All
    # our targets are 64-bit (the compiler default), so strip the flag for portability.
    cml = src / "CMakeLists.txt"
    cml.write_text(cml.read_text().replace("-m${BITNESS}", ""))

    build = src / "build"
    build.mkdir(exist_ok=True)
    cmake_args = ["cmake", "..", "-DCOOLPROP_SHARED_LIBRARY=ON", "-DCMAKE_BUILD_TYPE=Release"]
    # The bundled lib is dlopen'd at runtime (package data), so it is never a NEEDED of
    # the extension module and auditwheel never inspects it -- its libstdc++/libgcc deps
    # are not vendored or checked against the manylinux baseline. Static-link them so the
    # lib carries no host GLIBCXX/CXXABI requirement (clang/libc++ on macOS is ABI-stable,
    # and the MSVC runtime is handled separately, so this is Linux/gcc only).
    if sys.platform.startswith("linux"):
        cmake_args.append("-DCMAKE_SHARED_LINKER_FLAGS=-static-libstdc++ -static-libgcc")
    # match the wheel's target arch(es); cibuildwheel sets ARCHFLAGS on macOS
    archflags = os.environ.get("ARCHFLAGS", "")
    if sys.platform == "darwin":
        arches = [a for a in archflags.split() if a not in ("-arch", "")]
        if arches:
            cmake_args.append("-DCMAKE_OSX_ARCHITECTURES=" + ";".join(arches))
        # CoolProp uses std::filesystem (src/CPfilepaths.cpp), which Apple's libc++ marks
        # unavailable below macOS 10.15. CoolProp only pins CMAKE_OSX_DEPLOYMENT_TARGET under
        # -DDARWIN_USE_LIBCPP (unset here), so the build would otherwise inherit cibuildwheel's
        # lower default target and fail to compile. Force a >=10.15 floor -- this -D wins over
        # the MACOSX_DEPLOYMENT_TARGET env (cmake only falls back to the env when the cache var
        # is unset). Keep it in step with the wheel tag ([tool.cibuildwheel.macos] in
        # pyproject.toml); arm64 clamps up to its own 11.0 minimum.
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET") or "10.15"
        if tuple(int(p) for p in target.split(".")) < (10, 15):
            target = "10.15"
        cmake_args.append(f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}")
    if sys.platform == "win32":
        # Minimal Windows images (e.g. windows/servercore:ltsc2019) ship the OS UCRT but not
        # the VC++ redistributable, and CPython provides only vcruntime140.dll -- so the C++
        # CoolProp.dll's msvcp140.dll dependency is unmet there and the DLL fails to load.
        # Statically link the MSVC runtime into it (CoolProp's own /MT switch) so the wheel is
        # self-contained; the linker keeps only the runtime code actually used (roughly +1 MB on
        # a ~10 MB DLL), it does not embed the whole msvcp140.dll. The Rust plugin .pyd needs
        # only vcruntime140.dll, which ships with every CPython, so it stays on the dynamic CRT.
        cmake_args.append("-DCOOLPROP_MSVC_STATIC=ON")
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
