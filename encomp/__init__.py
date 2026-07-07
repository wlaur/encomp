"""Top-level package metadata for encomp."""

from importlib import metadata as _metadata

__all__ = ("__version__",)

try:
    __version__ = _metadata.version("encomp")
except _metadata.PackageNotFoundError:
    # no installed distribution metadata -- e.g. a source checkout, or the docs build,
    # which installs the dependencies but not the project itself (--no-install-project)
    __version__ = "0.0.0"
