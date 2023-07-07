# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)

autodoc_mock_imports = ["dolfin", "matplotlib", "ffc", "ufl", 
        "petsc4py", "mpi4py", "scipy", "numpy", "hippylib"]
autodoc_default_flags = ['members', 'private-members', 'undoc-members']
autoclass_content = 'both'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SOUPy'
copyright = "2023, Peng Chen, Dingcheng Luo, Thomas O'Leary-Roseberry, Umberto Villa"
author = "Peng Chen, Dingcheng Luo, Thomas O'Leary-Roseberry, Umberto Villa"
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode', 
              'sphinx.ext.napoleon', 'sphinx_mdinclude']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxdoc'
html_static_path = []
