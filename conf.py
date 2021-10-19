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
from recommonmark.parser import CommonMarkParser
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Carmen AI'
copyright = '2021, Nicholas Harris, Joshua Cooper, Garrett London, Guy Phelps, James Clabo, Cristian Henriquez'
author = 'Nicholas Harris, Joshua Cooper, Garrett London, Guy Phelps, James Clabo, Cristian Henriquez'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "myst_parser"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
html_theme = "alabaster"
html_static_path = ["_static"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None)
}
html_theme_options = {"nosidebar": True}

