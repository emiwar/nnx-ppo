"""Backwards-compatibility re-export shim.

Action samplers moved to :mod:`nnx_ppo.algorithms.distributions` — they are
policy-distribution machinery, not general network layers, and belong
alongside the rest of the PPO algorithm code.

New code should import from ``nnx_ppo.algorithms.distributions``. This
module re-exports the symbols and emits a ``DeprecationWarning`` on import
so callers in ``vnl-experiments`` keep working until they migrate.
"""

import warnings

from nnx_ppo.algorithms.distributions import ActionSampler, NormalTanhSampler

warnings.warn(
    "nnx_ppo.networks.sampling_layers has moved to "
    "nnx_ppo.algorithms.distributions. Update your imports; this shim will "
    "be removed once vnl-experiments are migrated to the new API.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ActionSampler", "NormalTanhSampler"]
