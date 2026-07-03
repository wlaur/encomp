import importlib.metadata

try:
    __version__ = importlib.metadata.version("encomp")
except importlib.metadata.PackageNotFoundError:
    # no installed distribution metadata -- e.g. a source checkout, or the docs build,
    # which installs the dependencies but not the project itself (--no-install-project)
    __version__ = "0.0.0"
