"""PPOAdapter: two-port router from network output to ``PPONetworkOutput``.

``PPOAdapter`` is a tiny :class:`StatefulModule` (~50 LoC) that takes two
sub-modules — an ``action`` port and a ``value`` port — runs both on its
upstream input, and packages their outputs into a
:class:`PPONetworkOutput`. It is the canonical way to make a ``Sequential``
trunk into a PPO-shaped network::

    pipeline = Sequential([
        Normalizer(obs_shape),
        trunk,                            # produces {action_params, value}
        PPOAdapter(
            action=Sequential([
                Filter({"action_params": "action_params"}),
                NormalTanhSampler(rngs, entropy_weight=1e-2),
            ]),
            value=Filter({"value": "value"}),
        ),
    ])

The action port's forward output must be a tree of *sampler dicts*. A
sampler dict is the small ``{"action", "log_likelihood"}`` payload each
:class:`ActionSampler` returns. For a single-sampler action port the
output IS a sampler dict; for a per-key sampler bank
(``Map({pop: sampler})``) the output is ``{pop: sampler_dict}``. The
adapter extracts fields uniformly via ``jax.tree.map`` with an
``is_leaf`` recogniser.

The value port's forward output is taken as-is for
``PPONetworkOutput.value_estimates``. Trailing singleton axes are
squeezed for ergonomic shape (``[B, 1]`` → ``[B]``).
"""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jp

from nnx_ppo.networks.types import (
    ModuleState,
    PPONetworkOutput,
    StatefulModule,
    StatefulModuleOutput,
)


_SAMPLER_DICT_KEYS = frozenset({"action", "log_likelihood"})


def _is_sampler_dict(x: Any) -> bool:
    return isinstance(x, dict) and _SAMPLER_DICT_KEYS.issubset(x.keys())


def _squeeze_trailing_one(v: Any) -> Any:
    if hasattr(v, "shape") and v.shape and v.shape[-1] == 1:
        return jp.squeeze(v, axis=-1)
    return v


class PPOAdapter(StatefulModule):
    """Two-port router producing :class:`PPONetworkOutput`.

    Args:
        action: The action port. Its forward output is a tree of sampler
            dicts ``{"action", "log_likelihood"}``.
        value: The value port. Its forward output is used directly as
            ``value_estimates`` (trailing singleton axes are squeezed).
    """

    def __init__(self, action: StatefulModule, value: StatefulModule):
        self.action = action
        self.value = value

    def __call__(
        self,
        state: dict[str, ModuleState],
        x: Any,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        if rollout_extras is None:
            a_re = v_re = None
        else:
            a_re = rollout_extras["action"]
            v_re = rollout_extras["value"]

        a_out = self.action(state["action"], x, a_re)
        v_out = self.value(state["value"], x, v_re)

        actions = jax.tree.map(
            lambda d: d["action"], a_out.output, is_leaf=_is_sampler_dict
        )
        loglikelihoods = jax.tree.map(
            lambda d: d["log_likelihood"],
            a_out.output,
            is_leaf=_is_sampler_dict,
        )
        value_estimates = jax.tree.map(_squeeze_trailing_one, v_out.output)

        return StatefulModuleOutput(
            next_state={
                "action": a_out.next_state,
                "value": v_out.next_state,
            },
            output=PPONetworkOutput(
                actions=actions,
                loglikelihoods=loglikelihoods,
                value_estimates=value_estimates,
            ),
            regularization_loss=a_out.regularization_loss
            + v_out.regularization_loss,
            metrics={"action": a_out.metrics, "value": v_out.metrics},
            rollout_extras={
                "action": a_out.rollout_extras,
                "value": v_out.rollout_extras,
            },
        )

    def initialize_state(self, batch_size: int) -> dict[str, ModuleState]:
        return {
            "action": self.action.initialize_state(batch_size),
            "value": self.value.initialize_state(batch_size),
        }

    def reset_state(self, prev_state: dict[str, ModuleState]) -> dict[str, ModuleState]:
        return {
            "action": self.action.reset_state(prev_state["action"]),
            "value": self.value.reset_state(prev_state["value"]),
        }

    def update_statistics(self, rollout_extras: Any) -> None:
        self.action.update_statistics(rollout_extras["action"])
        self.value.update_statistics(rollout_extras["value"])
