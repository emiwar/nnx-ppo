import dataclasses
from typing import Any, Union
import abc
import jax
from flax import nnx
from jaxtyping import Array, Float, PyTree

from nnx_ppo.jax_dataclass import JaxDataclass


ModuleState = PyTree  # Any JAX pytree: (), (h, c), dict, etc.


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PPONetworkOutput(JaxDataclass):
    """PPO-specific forward output. Produced by :class:`PPOAdapter`.

    Lives inside the ``output`` field of a :class:`StatefulModuleOutput`.
    ``regularization_loss`` and ``metrics`` flow through the enclosing
    ``StatefulModuleOutput`` — they are not duplicated here.
    """

    actions: Any
    loglikelihoods: PyTree[Float[Array, "*time batch"]]
    value_estimates: Any


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class StatefulModuleOutput(JaxDataclass):
    next_state: ModuleState
    output: Any
    regularization_loss: Float[Array, "*batch"]  # Scalar or batch-sized
    metrics: dict[Union[str, int], Any]
    rollout_extras: Any = None


class StatefulModule(abc.ABC, nnx.Module):
    """Abstract base class for network modules and layers, specifying the interface
    with the RL algorithm. Each module treated as stateful, with non-stateful modules
    having empty states.

    There are two types of module state. First, there is the state of the `nnx.Module`
    which follows the common NNX patterns. This state stores trainable parameters
    (of type `nnx.Param`) as well as any `RNGStreams` and other variables. This state
    is _not_ reset when the RL environment is reset, and **may not be written from
    inside** ``__call__`` if those writes affect the forward output.

    Stats-bearing modules (e.g. :class:`~nnx_ppo.networks.normalizer.Normalizer`)
    accumulate state by overriding :meth:`update_statistics`, which is called
    once per training step after the loss / gradient update.

    Second, there is an explicit carry state that is intended for stateful network
    layers, e.g. the hidden activations of RNNs. This state _is_ reset when the RL
    environment is reset.

    ``__call__`` takes ``rollout_extras`` as a third positional argument
    (default ``None``). It is a pytree shaped like the module's own contribution
    to the network's ROLLOUT → LOSS_REPLAY communication channel. In ROLLOUT the
    module emits its contribution as part of the returned ``StatefulModuleOutput``;
    in LOSS_REPLAY the same value is fed back in via this argument. Modules that
    don't need replay information leave it ``None``. The phase a module is in is
    derivable from this argument: ``None`` means ROLLOUT or INFERENCE
    (sample fresh / emit); a non-``None`` value means LOSS_REPLAY (consume).
    """

    @abc.abstractmethod
    def __call__(
        self,
        module_state: ModuleState,
        obs: PyTree,
        rollout_extras: Any = None,
    ) -> StatefulModuleOutput:
        """Args:
          module_state: The current state of the module.
          obs: Observation(s) to process. Leading dimension is batch.
          rollout_extras: Replay snapshots for this module (and, via container
            routing, its descendants). ``None`` in ROLLOUT / INFERENCE; the
            stored value from ``Transition.rollout_extras`` in LOSS_REPLAY.

        Returns ``StatefulModuleOutput`` with ``next_state``, ``output``,
        ``regularization_loss``, ``metrics``, and ``rollout_extras`` (the
        snapshot to be stored on the transition for later replay).
        """

    def initialize_state(self, batch_size: int) -> ModuleState:
        """Create a new state for this module.

        Args:
          batch_size: Batch size (leading dimension on any returned arrays).

        Returns:
          A ``ModuleState`` with ``batch_size`` as the leading dimension.
          Default is an empty tuple (stateless module).
        """
        return ()

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        return prev_state

    def update_statistics(self, rollout_extras: Any) -> None:
        """Fold the rollout's worth of replay snapshots into any running
        statistics this module owns. Called once per training step after
        the loss / gradient update.

        Containers override this to route ``rollout_extras`` per child the
        same way they route state. Stats-bearing leaves (e.g.
        :class:`Normalizer`) consume their ``[T, B, *feat]`` history slice
        and update their NNX variables in place. Default is a no-op.
        """
        del rollout_extras
        return None
