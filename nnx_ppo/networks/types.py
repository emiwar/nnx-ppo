import dataclasses
import enum
from typing import Optional, Any, Union, TYPE_CHECKING
from warnings import deprecated
import abc
import jax
from flax import nnx
from jaxtyping import Array, Float, PyTree, ScalarLike

if TYPE_CHECKING:
    from nnx_ppo.algorithms.types import Transition

from nnx_ppo.jax_dataclass import JaxDataclass


class Context(enum.Enum):
    """The lifecycle context in which a network forward pass is being made.

    Threaded through every ``StatefulModule.__call__`` as a keyword-only
    argument. Each module is free to behave differently per context.

    - ``ROLLOUT``: collecting training data by driving the environment.
      Action samplers sample stochastically.
    - ``LOSS_REPLAY``: re-running the rollout to compute gradients. Action
      samplers use the stored ``raw_action``. Forward output must be
      deterministic with respect to rollout's RNG state.
    - ``INFERENCE``: evaluation, distillation teacher, ad-hoc forward
      passes. Sampler's per-instance ``deterministic`` flag decides
      stochastic-vs-mean behaviour.
    - ``STATS_UPDATE``: post-loss replay of the rollout. The only context
      in which modules may write to NNX variables that affect future
      forward output (e.g. ``Normalizer`` mean/M2/counter).

    Default for ``__call__`` is ``INFERENCE`` — the safe choice. If a
    caller forgets to pass ``context=``, nothing surprising happens.
    """
    ROLLOUT = "rollout"
    LOSS_REPLAY = "loss_replay"
    INFERENCE = "inference"
    STATS_UPDATE = "stats_update"


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PPONetworkOutput(JaxDataclass):
    # actions and raw_actions are Any (not Float[Array, ...]) to support dict/PyTree
    # actions for multi-agent or modular networks. See PPONetwork.__call__ docstring.
    actions: Any
    raw_actions: Any
    loglikelihoods: PyTree[Float[Array, "*time batch"]]
    regularization_loss: Float[Array, "..."]  # Scalar or batch-sized; broadcastable
    value_estimates: Any  # Float[Array, "*time batch"] or dict thereof for multi-reward networks
    metrics: dict[str, Any]
    # Pre-sampling distribution parameters keyed by action name. Populated by
    # adapters (e.g. PPOAdapter) so distillation losses can read student and
    # teacher distribution params without re-running the network. None when
    # the network is not wrapped in a distribution-aware adapter.
    distribution_params: Any = None


ModuleState = PyTree  # Any JAX pytree: (), (h, c), dict, etc.


class PPONetwork(nnx.Module, abc.ABC):

    @abc.abstractmethod
    def __call__(
        self,
        network_state: ModuleState,
        obs: PyTree,
        raw_action: Optional[Any] = None,
        *,
        context: Context = Context.INFERENCE,
    ) -> tuple[ModuleState, PPONetworkOutput]:
        """Apply both actor and critic networks to the environment observation `obs`.

        Calling the critic on rollouts might be somewhat inefficient, but by grouping
        them into the same method, the interface is well-defined even if the actor and
        critic share stateful layers.

        Args:
          network_state (ModuleState): The activation state of the network plus any RNG keys
                                       used for stochastic layers.
          obs (PyTree): Observation of an environment state. It is a PyTree with the same
                        structure as env.step(...).obs
          raw_action (jax.Array): If specified, compute likelihoods using these actions
                                  than sampling a new action. `raw_action` referes to
                                  the sampled action before clamping (e.g., by tanh).
          context (Context): The lifecycle context. Threaded to all child modules.

        Returns: (new_state, network_output)
          new_state (PyTree): An updated network state, including split RNG keys.
          network_output (PPONetworkOutput):
            actions (PyTree): The actions to be sent back to the env
            raw_actions (PyTree): The raw output of the sampler before applying clamping
            loglikelihood (float): The log-likelihood for taking the action
            regularization_loss (float): Sum of all other regularizer terms. This will be added
                                 to the RL loss target during training.
            value_estimates (float): The estimated value of `obs` by the critic network.
            metrics (dict): A dictionary with any intermediate values to log

        Note: for multi-agent / modular networks, the returned action, log likelihood, and
              and value estimate may be arbitrary PyTrees instead of floats.
        """

    @abc.abstractmethod
    def initialize_state(self, batch_size: int) -> ModuleState:
        """Create an initial state for the network, including RNG keys for any
        stochastic layers.

        Args:
          batch_size (int): The batch size of the returned state.
        """

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        """Specifies how the network should be transformed when the corresponding
        environment is reset.

        Args:
          prev_state: The state of the network before the environment was reset
        """
        return prev_state

    @deprecated(
        "Use the context=Context.STATS_UPDATE forward pass to update running "
        "statistics inside __call__ instead. This hook remains for backwards "
        "compatibility with networks that haven't migrated yet and will be "
        "removed once vnl-experiments are rewritten on the new API."
    )
    def update_statistics(
        self, last_rollout: "Transition", total_steps: ScalarLike
    ) -> None:
        return None


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class StatefulModuleOutput(JaxDataclass):
    next_state: ModuleState
    output: Any
    regularization_loss: Float[Array, "*batch"]  # Scalar or batch-sized
    metrics: dict[
        Union[str, int], Any
    ]  # Keys can be str or int, values are arrays or nested dicts


class StatefulModule(abc.ABC, nnx.Module):
    """Abstract base class for network modules and layers, specifying the interface
    with the RL algorithm. Each module treated as stateful, with non-stateful modules
    having empty states.

    There are two types of module state. First, there is the state of the `nnx.Module`
    which follows the common NNX patterns. This state stores trainable parameters
    (of type `nnx.Param`) as well as any `RNGStreams` and other variables. This state
    is _not_ reset when the RL environment is reset.

    Second, there is an explicit carry state that is intended for stateful network
    layers, e.g. the hidden activations of RNNs. This state _is_ reset when the RL
    environment is reset.

    ``__call__`` takes a keyword-only ``context: Context`` argument. Modules whose
    behavior varies by context (``Normalizer``, ``ActionSampler``, ...) read this
    argument; containers thread it to children. The default is ``Context.INFERENCE``,
    which is the safe choice — no NNX variable writes that affect forward output.
    Writes to such variables are only permitted when ``context == Context.STATS_UPDATE``.
    """

    @abc.abstractmethod
    def __call__(
        self,
        module_state: ModuleState,
        obs: PyTree,
        *,
        context: Context = Context.INFERENCE,
    ) -> StatefulModuleOutput:
        """Args:
          module_state (ModuleState):
            The current state of the module.
          obs (PyTree):
            Observation(s) to process. Leading dimension is batch.
          context (Context):
            Lifecycle context. Containers thread this to children; modules that
            care key off of it.
        Returns (StatefulModuleOutput):
          next_state (ModuleState):
            The updated module state after processing `obs`. Should have the same
            structure as `module_state`.
          output (PyTree):
            The output of the network module. Should have batch as leading dimension.
          regularization_loss (float):
            A floating-point value that is added to the total loss. Useful for
            adding regularization or co-objectives to the networks.
          metrics (Dict[str, jax.Array]):
            Any logging metrics.
        """

    def initialize_state(self, batch_size: int) -> ModuleState:
        """Create a new state for this module.

        Args:
        batch_size (int): The batch size

        Returns:
        A `ModuleState` with `batch_size` as the leading dimension on any
        arrays in the tree. The default is an empty tuple, representing a
        stateless module.

        Notes:
        As JAX does not allow parallellized conditional evaluation, this
        method will be called on every step. For environments where `done=True`,
        the corresponding network state will be replaced with the return value of
        this method.
        """
        return ()

    def reset_state(self, prev_state: ModuleState) -> ModuleState:
        return prev_state

    @deprecated(
        "Use context=Context.STATS_UPDATE in __call__ instead. This hook is "
        "kept only for backwards compatibility with networks that haven't "
        "migrated to the context-based API and will be removed once the "
        "experiment rewrites land."
    )
    def update_statistics(
        self, last_rollout: "Transition", total_steps: ScalarLike
    ) -> None:
        """Legacy hook called by older training code after a rollout. New code
        relies on the per-step inline updates that ``Normalizer`` and other
        stats-bearing modules perform when ``__call__`` is invoked with
        ``context=Context.STATS_UPDATE``.

        Args:
          last_rollout (Transition[T, B, ...]): dataclass with the most recent rollout.
          total_steps: Total number of steps taken so far.
        """
        return None
