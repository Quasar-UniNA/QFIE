import os
import sys

#sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("../../src/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QFIE_Package'
copyright = '2022, Roberto Schiattarella'
author = 'Roberto Schiattarella'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
  'sphinx.ext.viewcode',
  'sphinx.ext.napoleon',
  'nbsphinx',
  'IPython.sphinxext.ipython_console_highlighting']

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']

html_show_sourcelink = True
