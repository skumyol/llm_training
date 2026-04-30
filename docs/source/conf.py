# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
project = "NPC Social State Dialogue"
copyright = "2024, NPC Dialogue Research"
author = "NPC Dialogue Research Team"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",               # Markdown support
    "sphinx.ext.autodoc",        # Auto-doc from docstrings
    "sphinx.ext.napoleon",       # Google/NumPy style docstrings
    "sphinx.ext.viewcode",       # Add links to source code
    "sphinx.ext.intersphinx",    # Link to external docs
    "sphinx.ext.todo",           # TODO directives
    "sphinx_copybutton",         # Copy button on code blocks
]

# MyST parser config
myst_enable_extensions = [
    "colon_fence",    # ::: fences for code blocks
    "deflist",        # Definition lists
    "dollarmath",     # $...$ inline math
    "amsmath",        # $$...$$ block math
    "fieldlist",      # Field lists
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# -- Path setup --------------------------------------------------------------
# Add source directories to path for autodoc
sys.path.insert(0, os.path.abspath("../llm_finetuning/src"))
sys.path.insert(0, os.path.abspath("../slm_training/src"))

# Suppress autodoc warnings for external modules
autodoc_mock_imports = [
    "torch", "torch.nn", "torch.nn.functional",
    "transformers", "peft", "bitsandbytes",
    "mlflow", "yaml", "sklearn", "tqdm",
    "numpy", "pandas", "optuna",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "NPC Social State Dialogue"
html_short_title = "NPC Dialogue"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "font-stack": "system-ui, -apple-system, sans-serif",
        "font-stack--monospace": "Consolas, 'Liberation Mono', monospace",
    },
}

# -- Extension configuration -------------------------------------------------
todo_include_todos = True

# -- ReadTheDocs -------------------------------------------------------------
# These are set automatically by ReadTheDocs
# https://docs.readthedocs.io/en/stable/config-file/v2.html
