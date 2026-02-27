"""Checkpointing utilities for saving and loading training state."""

import os
import pickle
from collections.abc import Callable
from typing import Any, Optional, Protocol, runtime_checkable

import jax
from flax import nnx

from nnx_ppo.algorithms.config import TrainConfig
from nnx_ppo.algorithms.types import TrainingState


@runtime_checkable
class CheckpointCallback(Protocol):
    """Protocol for checkpoint callbacks with named parameters."""

    def __call__(self, training_state: TrainingState, step: int) -> None: ...


def _split_net_state(networks):
    """Split network state: RngKey → pickle, everything else → orbax.

    orbax cannot handle JAX new-style PRNG key arrays (dtype ``key<fry>``), so
    we separate nnx.RngKey variables and persist them with pickle instead. All
    other variable types — including nnx.Param, nnx.RngCount, and custom
    Variable subclasses such as NormalizerStatistics — are saved via orbax.

    Returns:
        (non_key_state, rng_key_state, abstract_non_key) — the first two are
        nnx.State objects and the third is the abstract (ShapeDtypeStruct)
        target needed for orbax restoration.
    """
    _, rng_key_state, non_key_state = nnx.split(networks, nnx.RngKey, ...)
    abstract_non_key = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), non_key_state
    )
    return non_key_state, rng_key_state, abstract_non_key


def make_checkpoint_fn(
    directory: str,
    config: Optional[TrainConfig] = None,
) -> CheckpointCallback:
    """Create a checkpoint callback that saves TrainingState to disk.

    Each checkpoint is written to ``{directory}/step_{step:010d}/``, containing:

    - ``networks/`` — orbax checkpoint with all non-PRNG-key network variables
      (Param, RngCount, NormalizerStatistics, etc.)
    - ``optimizer/`` — orbax checkpoint with all optimizer state arrays
    - ``metadata.pkl`` — pickle file with network RngKey variables,
      all remaining TrainingState fields (``network_states``, ``env_states``,
      ``rng_key``, ``steps_taken``), the step count, and the optional
      TrainConfig.

    To resume training from a checkpoint, use :func:`load_checkpoint`.

    Args:
        directory: Base directory under which checkpoint subdirectories are
            created.
        config: Optional TrainConfig to store alongside each checkpoint, useful
            for reproducing training runs.

    Returns:
        A callback compatible with train_ppo's ``checkpoint_fn`` parameter.

    Example:
        >>> result = train_ppo(
        ...     env, networks, config,
        ...     checkpoint_fn=make_checkpoint_fn("/tmp/my_run", config=config),
        ... )
    """

    abs_directory = os.path.abspath(directory)

    def checkpoint_fn(training_state: TrainingState, step: int) -> None:
        import orbax.checkpoint as ocp

        step_dir = os.path.join(abs_directory, f"step_{step:010d}")
        os.makedirs(step_dir, exist_ok=True)

        # Split network state: everything except RngKey → orbax, RngKey → pickle.
        # orbax cannot handle JAX new-style PRNG key arrays.
        non_key_state, rng_key_state, _ = _split_net_state(training_state.networks)

        # The optimizer only contains float/int arrays; no key arrays.
        _, opt_state = nnx.split(training_state.optimizer)

        # Save parameter arrays with orbax. A fresh checkpointer is created per
        # call and immediately closed to ensure all async writes complete.
        checkpointer = ocp.StandardCheckpointer()
        try:
            checkpointer.save(os.path.join(step_dir, "networks"), non_key_state)
            checkpointer.save(os.path.join(step_dir, "optimizer"), opt_state)
        finally:
            checkpointer.close()

        # Save everything else with pickle (JAX arrays including PRNG keys are
        # pickle-safe).
        metadata = {
            "networks_rng_key_state": rng_key_state,
            "network_states": training_state.network_states,
            "env_states": training_state.env_states,
            "rng_key": training_state.rng_key,
            "steps_taken": training_state.steps_taken,
            "step": step,
            "config": config,
        }
        with open(os.path.join(step_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    return checkpoint_fn


def load_checkpoint(
    path: str,
    networks: Any,
    optimizer: nnx.Optimizer,
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`make_checkpoint_fn`.

    The ``networks`` and ``optimizer`` arguments serve as structural templates:
    their architecture must match the checkpoint, but their current parameter
    values are irrelevant and will be overwritten in-place by the checkpoint
    values.

    Args:
        path: Path to the step checkpoint directory, e.g.
            ``/tmp/my_run/step_0000500000``.
        networks: Network instance with the same architecture as the checkpoint.
            Weights are updated in-place.
        optimizer: Optimizer instance with the same structure as the checkpoint.
            State is updated in-place.

    Returns:
        A dict with the following keys:

        - ``"training_state"`` — restored :class:`TrainingState`
        - ``"step"`` — training step at which the checkpoint was saved (int)
        - ``"config"`` — :class:`TrainConfig` if one was stored, else ``None``

    Example:
        >>> networks = factories.make_mlp_actor_critic(...)
        >>> training_state = ppo.new_training_state(env, networks, n_envs, seed)
        >>> ckpt = load_checkpoint(
        ...     "/tmp/my_run/step_0000500000",
        ...     training_state.networks,
        ...     training_state.optimizer,
        ... )
        >>> result = train_ppo(
        ...     env, networks, ckpt["config"],
        ...     initial_state=ckpt["training_state"],
        ... )
    """
    import orbax.checkpoint as ocp

    path = os.path.abspath(path)

    # Build abstract targets from the user-provided templates.
    # Use ... to capture remaining variables (RngKey) that we restore via pickle.
    _, _, abstract_non_key = nnx.split(networks, nnx.RngKey, ...)
    abstract_non_key = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), abstract_non_key
    )
    _, opt_template = nnx.split(optimizer)
    opt_abstract = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), opt_template
    )

    checkpointer = ocp.StandardCheckpointer()
    try:
        restored_non_key = checkpointer.restore(
            os.path.join(path, "networks"), abstract_non_key
        )
        restored_opt = checkpointer.restore(
            os.path.join(path, "optimizer"), opt_abstract
        )
    finally:
        checkpointer.close()

    with open(os.path.join(path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    # Merge orbax-restored non-key state with pickled rng-key state,
    # then update the provided modules in-place.
    full_net_state = nnx.merge_state(restored_non_key, metadata["networks_rng_key_state"])
    nnx.update(networks, full_net_state)
    nnx.update(optimizer, restored_opt)

    training_state = TrainingState(
        networks=networks,
        network_states=metadata["network_states"],
        env_states=metadata["env_states"],
        optimizer=optimizer,
        rng_key=metadata["rng_key"],
        steps_taken=metadata["steps_taken"],
    )
    return {
        "training_state": training_state,
        "step": metadata["step"],
        "config": metadata["config"],
    }