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

# Make a symbolic link to the tutorials directory. Quick hack to include 
# Jupyter notebooks easily 
os.system("rm tutorials")
os.system("ln -s ../tutorials")


autodoc_mock_imports = ["dolfin", "matplotlib", "ffc", "ufl", 
        "petsc4py", "mpi4py", "scipy", "numpy", "hippylib"]
autodoc_default_flags = ['members', 'private-members', 'undoc-members']
autoclass_content = 'both'

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SOUPy'
copyright = "2023, Peng Chen, Dingcheng Luo, Thomas O'Leary-Roseberry, Umberto Villa, Omar Ghattas"
author = "Peng Chen, Dingcheng Luo, Thomas O'Leary-Roseberry, Umberto Villa, Omar Ghattas"
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode', 
              'sphinx.ext.napoleon', 'myst_nb', 'sphinx.ext.autosummary']


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}
master_doc = 'index'

nb_execution_mode = "off" 
myst_heading_anchors = 3 
myst_enable_extensions = ['amsmath', 'colon_fence', 'deflist', 'dollarmath', 'html_image']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
# html_theme_options = {
    # "nosidebar": True
# }
# html_theme_options = {
#     "nosidebar": False, "sidebarwidth": "40em"
# }
