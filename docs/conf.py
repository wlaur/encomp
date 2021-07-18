# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from encomp import __version__


# -- Project information -----------------------------------------------------

project = 'encomp'
copyright = '2021, William Laurén'
author = 'William Laurén'


release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# add mock imports to avoid running modules:
autodoc_mock_imports = ['encomp.notebook', 'encomp.magics']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_css_files = [
    'custom.css',
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

# clean theme with a lot of nice interactive content with Bootstrap
# blue in sidebar: 0, 91, 129
html_theme = 'pydata_sphinx_theme'

html_logo = 'img/logo.svg'
html_favicon = 'img/favicon.ico'

html_sidebars = {'**': ['globaltoc.html', 'relations.html',
                        'sourcelink.html', 'searchbox.html']}

autosummary_generate = True

todo_include_todos = True

pygments_style = 'sphinx'

# the images must be included in the _build dir when generating the HTML docs
# make sure they are referenced inside the corresponding notebook
# NOTE: the .ipynb suffix is not included in the names
nbsphinx_thumbnails = {
    # 'notebooks/test': '_images/test-thumb.png',

}

add_module_names = False
html_scaled_image_link = False


# show type hints in doc body instead of signature
autodoc_typehints = 'description'
autoclass_content = 'both'  # get docstring from class level and init simultaneously

nbsphinx_allow_errors = True


from sphinx.domains.python import PythonDomain

class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if 'refspecific' in node:
            del node['refspecific']
        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)

def setup(sphinx):
    sphinx.add_domain(PatchedPythonDomain, override=True)
