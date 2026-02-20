"""Pytest configuration for nnx_ppo tests.

Enables jaxtyping runtime shape checking via beartype during test runs.
"""

from jaxtyping import install_import_hook

# Enable runtime shape checking for all nnx_ppo modules during tests.
# This validates that array shapes match jaxtyping annotations.
install_import_hook("nnx_ppo", "beartype.beartype")
