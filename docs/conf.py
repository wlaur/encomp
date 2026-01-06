# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.


import sys
from pathlib import Path

from sphinx.domains.python import PythonDomain

# make sure the local source is loaded when importing
sys.path.insert(0, str(Path("..").absolute()))
from encomp import __version__

# -- Project information -----------------------------------------------------

project = "encomp"
copyright = "2026, William Laurén"
author = "William Laurén"


release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
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
html_static_path = ["_static"]


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

todo_include_todos = True

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# the images must be included in the _build dir when generating the HTML docs
# make sure they are referenced inside the corresponding notebook
# NOTE: the .ipynb suffix is not included in the names
nbsphinx_thumbnails = {
    "notebooks/test": "_images/logo.svg",
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
]

nbsphinx_allow_errors = True

nbsphinx_requirejs_path = ""
nbsphinx_requirejs_options = {}


class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if "refspecific" in node:
            del node["refspecific"]
        return super().resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode
        )


def setup(sphinx) -> None:
    sphinx.add_domain(PatchedPythonDomain, override=True)
