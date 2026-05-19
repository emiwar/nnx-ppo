"""PPOAdapter: thin PPONetwork wrapping an arbitrary StatefulModule.

Replaces ``PPOActorCritic`` in [containers.py](../networks/containers.py).
Where ``PPOActorCritic`` hardcoded the actor / critic / sampler / preprocessor
decomposition, ``PPOAdapter`` wraps any ``StatefulModule`` whose forward
output is a dict of named heads. The user declares which heads are actions
(and what sampler each goes through) and which heads are value estimates;
the adapter wires the rest.

This generalises naturally to:

- Single-head actor-critic (one action sampler, one value head).
- Modular networks (many action samplers, many value heads).
- Shared trunks (action heads and value heads both fed by the same upstream
  features inside the inner module).

The adapter populates ``PPONetworkOutput.distribution_params`` with each
sampler's metrics dict (mean / sigma for ``NormalTanhSampler``), so
distillation losses can read student and teacher distribution parameters
without a separate forward pass.
"""

from typing import Any, Optional, Union
from collections.abc import Sequence

import jax.numpy as jp
from flax import nnx

from nnx_ppo.algorithms.distributions import ActionSampler
from nnx_ppo.networks.types import (
    Context,
    ModuleState,
    PPONetwork,
    PPONetworkOutput,
    StatefulModule,
)


class PPOAdapter(PPONetwork):
    """Wraps a ``StatefulModule`` whose output is a dict of named heads.

    Args:
        inner: The wrapped module. ``inner(state, obs, context=...)`` must
            return a ``StatefulModuleOutput`` whose ``.output`` is a dict
            containing keys matching every entry of ``action_specs`` and
            every name in ``value_specs``.
        action_specs: Maps inner-output name → ``ActionSampler``. The
            sampler is called on ``inner_output[name]`` (typically a
            ``[batch, 2*action_dim]`` mean/log_std concat).
        value_specs: Either a single string naming the value head, or a
            sequence / dict whose keys name multiple value heads. Each
            named head's output is read out and squeezed to remove a
            trailing singleton axis (so ``shape=[batch, 1]`` becomes
            ``[batch]``).

    Behaviour with a single action / value (``len(action_specs) == 1``,
    ``value_specs`` is a string or has one entry): the adapter unwraps the
    single-entry dicts into raw arrays, so ``output.actions``,
    ``output.raw_actions``, ``output.loglikelihoods``, and
    ``output.value_estimates`` are arrays (matching the legacy
    ``PPOActorCritic`` shape). With multiple entries they stay as dicts
    keyed by the spec names.
    """

    def __init__(
        self,
        inner: StatefulModule,
        action_specs: dict[str, ActionSampler],
        value_specs: Union[str, Sequence[str], dict[str, Any]],
    ):
        if not action_specs:
            raise ValueError("PPOAdapter requires at least one action spec")

        if isinstance(value_specs, str):
            value_names = [value_specs]
        elif isinstance(value_specs, dict):
            value_names = list(value_specs.keys())
        else:
            value_names = list(value_specs)
        if not value_names:
            raise ValueError("PPOAdapter requires at least one value spec")

        self.inner = inner
        self.action_samplers = nnx.Dict(action_specs)
        # Plain Python state; not NNX-tracked.
        self._action_keys = list(action_specs.keys())
        self._value_names = value_names
        self._single_action = len(self._action_keys) == 1
        self._single_value = len(value_names) == 1

    def __call__(
        self,
        network_state: dict[str, Any],
        obs: Any,
        raw_action: Optional[Any] = None,
        *,
        context: Context = Context.INFERENCE,
    ) -> tuple[dict[str, Any], PPONetworkOutput]:
        inner_out = self.inner(network_state["inner"], obs, context=context)
        inner_dict = inner_out.output

        # Normalise raw_action to a dict keyed by sampler name.
        if raw_action is None:
            ra_per_sampler: dict[str, Optional[Any]] = {
                k: None for k in self._action_keys
            }
        elif self._single_action and not isinstance(raw_action, dict):
            ra_per_sampler = {self._action_keys[0]: raw_action}
        else:
            ra_per_sampler = raw_action

        new_sampler_states: dict[str, Any] = {}
        actions: dict[str, Any] = {}
        raw_actions: dict[str, Any] = {}
        loglikelihoods: dict[str, Any] = {}
        distribution_params: dict[str, Any] = {}
        sampler_metrics: dict[str, Any] = {}
        reg_loss = inner_out.regularization_loss

        for name, sampler in self.action_samplers.items():
            head_params = inner_dict[name]
            sampler_out = sampler(
                network_state["samplers"][name],
                head_params,
                ra_per_sampler[name],
                context=context,
            )
            new_sampler_states[name] = sampler_out.next_state
            a, r, l = sampler_out.output
            actions[name] = a
            raw_actions[name] = r
            loglikelihoods[name] = l
            # ActionSampler stashes distribution params (mu/sigma for
            # NormalTanh) in its metrics dict; surface them on the output.
            distribution_params[name] = sampler_out.metrics
            sampler_metrics[name] = sampler_out.metrics
            reg_loss = reg_loss + sampler_out.regularization_loss

        def _squeeze_value(v):
            if hasattr(v, "shape") and v.shape and v.shape[-1] == 1:
                return jp.squeeze(v, axis=-1)
            return v

        values: dict[str, Any] = {
            n: _squeeze_value(inner_dict[n]) for n in self._value_names
        }

        if self._single_action:
            k = self._action_keys[0]
            actions = actions[k]
            raw_actions = raw_actions[k]
            loglikelihoods = loglikelihoods[k]
        if self._single_value:
            values = values[self._value_names[0]]

        new_state = {
            "inner": inner_out.next_state,
            "samplers": new_sampler_states,
        }

        return new_state, PPONetworkOutput(
            actions=actions,
            raw_actions=raw_actions,
            loglikelihoods=loglikelihoods,
            regularization_loss=reg_loss,
            value_estimates=values,
            metrics={"inner": inner_out.metrics, "samplers": sampler_metrics},
            distribution_params=distribution_params,
        )

    def initialize_state(self, batch_size: int) -> dict[str, Any]:
        return {
            "inner": self.inner.initialize_state(batch_size),
            "samplers": {
                name: sampler.initialize_state(batch_size)
                for name, sampler in self.action_samplers.items()
            },
        }

    def reset_state(self, prev_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "inner": self.inner.reset_state(prev_state["inner"]),
            "samplers": {
                name: sampler.reset_state(prev_state["samplers"][name])
                for name, sampler in self.action_samplers.items()
            },
        }

    def update_statistics(self, last_rollout, total_steps) -> None:
        """Backwards-compat dispatch to children's legacy update_statistics.

        Walks ``self.inner`` and each action sampler, letting any module
        that hasn't migrated to the ``context=STATS_UPDATE`` forward path
        update its statistics through its own override. Currently this is
        how ``Normalizer`` is kept up to date — its inline STATS_UPDATE
        path will replace this once ``ppo_step`` switches to the
        STATS_UPDATE replay pass.
        """
        self.inner.update_statistics(last_rollout, total_steps)
        for sampler in self.action_samplers.values():
            sampler.update_statistics(last_rollout, total_steps)
