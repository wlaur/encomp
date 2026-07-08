# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import os
import sys
import tomllib
from pathlib import Path

from sphinx.domains.python import PythonDomain

# make sure the local source is loaded when importing (autodoc imports encomp from source;
# the project itself is not installed on RTD -- see .readthedocs.yaml --no-install-project)
sys.path.insert(0, str(Path("..").absolute()))

# -- Project information -----------------------------------------------------

project = "encomp"
copyright = "2026, William Laurén"
author = "William Laurén"


release = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())["project"]["version"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # Markdown source (the docs are Markdown, not reStructuredText)
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx_inline_tabs",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

# The documentation is written in Markdown (parsed by MyST); the README at the repo root is
# the single source of truth and is included verbatim into the landing page. nbsphinx still
# registers the .ipynb suffix for the example notebooks.
source_suffix = {
    ".md": "markdown",
}

# MyST features used by the docs: ::: colon fences (so autodoc directives can live in .md via
# {eval-rst}), $...$ / $$...$$ math, definition/field lists, and raw HTML <img> tags (the
# README logo needs an <img> tag to control its rendered size on GitHub/PyPI).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "dollarmath",
    "substitution",
    "html_image",
]

# generate anchors for headings h1-h3 so in-page Markdown links like
# [...](#parallel-coolprop-evaluation-with-polars) resolve
myst_heading_anchors = 3

# prefix section labels with the document name so the README (included into index.md) and the
# usage guide can share section titles ("The Fluid class", ...) without duplicate-label warnings
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# list of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {"sidebar_hide_name": True}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# img/ is included so the logo is available in the build output (the notebook
# gallery thumbnail below references it; nbsphinx does not copy thumbnail
# files that are not part of a document)
html_static_path = ["_static", "img"]


html_css_files = [
    "custom.css",
]


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# blue in logo: 0, 91, 129
html_theme = "furo"

html_logo = "img/logo-small.svg"
html_favicon = "img/favicon.ico"

autosummary_generate = True
# Generate summaries only for documented items (reduces stub warnings)
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Disable numpydoc autosummary to avoid stub file warnings
numpydoc_show_class_members = False

todo_include_todos = False

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# gallery thumbnails for the example notebooks; the values must exist in the build
# output -- _static/logo.svg comes from the img/ entry in html_static_path above
# NOTE: the .ipynb suffix is not included in the names
nbsphinx_thumbnails = {
    "notebooks/getting-started": "_static/logo.svg",
}

add_module_names = False
html_scaled_image_link = False

# show type hints in doc body instead of signature
autodoc_typehints = "description"
autoclass_content = "both"  # get docstring from class level and init simultaneously

# Suppress warnings for forward references in type hints and missing autosummary stubs
suppress_warnings = [
    "sphinx_autodoc_typehints",  # Suppress all typehints warnings including forward refs
    "autosummary",  # Suppress missing stub file warnings
    # Two docutils errors come from autodoc rendering, not from the Markdown docs, and have no
    # clean source fix: (1) the inherited sympy.Symbol.__init__ docstring has an inconsistent
    # literal block, and (2) the MT_ / DT_ type variables end in an underscore, which docutils
    # reads as a hyperlink reference ("Unknown target name: mt") when the type hints are rendered
    # into the description. The Markdown sources themselves produce no docutils warnings, so
    # suppressing this category only hides these autodoc-internal glitches.
    "docutils",
]

# Execute the example notebooks where the compiled plugin is importable (local dev and
# CI, which run after `maturin develop`). On ReadTheDocs the project is installed with
# --no-install-project (.readthedocs.yaml) and nbsphinx runs each notebook in a separate
# Jupyter kernel that inherits neither conf.py's sys.path insert nor the compiled plugin,
# so `import encomp` fails there -- render the committed cell outputs instead. CI's
# `sphinx-build -W` remains the notebook-correctness gate. See CoolProp plugin notes in
# encomp/coolprop/README.md and the getting-started notebook's committed outputs.
nbsphinx_execute = "never" if os.environ.get("READTHEDOCS") else "always"
nbsphinx_allow_errors = False

nbsphinx_requirejs_path = ""
nbsphinx_requirejs_options = {}


class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if "refspecific" in node:
            del node["refspecific"]
        return super().resolve_xref(env, fromdocname, builder, typ, target, node, contnode)


def setup(sphinx) -> None:
    sphinx.add_domain(PatchedPythonDomain, override=True)
