# conf.py

import os
import sys

# Include the project's module path
sys.path.insert(0, os.path.abspath('../../module'))

# Project information
project = 'LLY-DML'
author = 'Your Name'
version = '1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google or NumPy style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
