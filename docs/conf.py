import os
import sys
sys.path.insert(0, os.path.abspath('../../module'))

# -- Project information -----------------------------------------------------
project = 'LLY-DML'
author = 'Your Name'
release = '1.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
