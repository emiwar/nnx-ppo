"""Sphinx configuration for nnx-ppo documentation."""

import os
import sys

# Make the package importable without installation.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "nnx-ppo"
author = "Emil Wärnberg"
copyright = "2024, Emil Wärnberg"
release = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# reward_scaling_wrapper.py imports mujoco_playground at the top level;
# mock it so the Sphinx build doesn't require the package to be installed.
autodoc_mock_imports = [
    "mujoco_playground",
    "brax",
    "wandb",
    "orbax",
    "ml_collections",
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__call__, __init__",
    "show-inheritance": True,
}

# Preserve the order in which members appear in source files.
autodoc_member_order = "bysource"

# Put verbose jaxtyping annotations (e.g. Float[Array, "batch features"])
# in the description section rather than the signature line.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Napoleon (Google-style docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_rtype = True

# ---------------------------------------------------------------------------
# Intersphinx — cross-links to external package docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
}

# ---------------------------------------------------------------------------
# HTML output — furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = "nnx-ppo"
