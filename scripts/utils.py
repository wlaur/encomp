import os
import platform

try:
    import fire
except ImportError:
    print('This script requires the "fire" package: pip install fire')
    fire = None


def build_docs() -> None:
    if platform.system() == "Windows":
        cmds = [
            # exclude the encomp.tests subpackage
            "sphinx-apidoc -f -o docs/source encomp encomp/tests",
            "call docs/make clean",
            "call docs/make html",
        ]

    else:
        cmds = [
            # exclude the encomp.tests subpackage
            "sphinx-apidoc -f -o docs/source encomp encomp/tests",
            "make --directory docs clean",
            "make --directory docs html",
        ]

    for n in cmds:
        os.system(n)


def pypi_upload() -> None:
    cmds = ["twine upload dist/*"]

    for n in cmds:
        os.system(n)


def main(task: str | None = None) -> None:
    if task == "pypi":
        pypi_upload()

    elif task == "docs":
        build_docs()

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    fire.Fire(main)
