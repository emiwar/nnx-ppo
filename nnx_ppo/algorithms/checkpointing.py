"""Checkpointing utilities for saving and loading training state."""

import os
import pickle
import shutil
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


#: Prefix of the directory a checkpoint is assembled in before being renamed
#: into place. Deliberately does not start with ``step_`` (and is hidden), so
#: that neither :func:`latest_checkpoint` nor a caller's own ``step_*`` glob can
#: mistake a half-written checkpoint for a complete one.
_TMP_PREFIX = ".tmp-"


def make_checkpoint_fn(
    directory: str,
    config: Optional[TrainConfig] = None,
    *,
    include_env_state: bool = True,
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

    The checkpoint is assembled in a temporary directory and renamed into place,
    so a process killed mid-write leaves no directory that looks like a complete
    checkpoint.

    To resume training from a checkpoint, use :func:`load_checkpoint`.

    Args:
        directory: Base directory under which checkpoint subdirectories are
            created.
        config: Optional TrainConfig to store alongside each checkpoint, useful
            for reproducing training runs.
        include_env_state: Whether to pickle ``env_states`` and
            ``network_states``. They are what makes a checkpoint bit-exactly
            resumable, and for environments with a large per-env state they can
            dominate its size and write time. Pass False for a *light*
            checkpoint (weights, optimizer, ``rng_key`` and ``steps_taken``
            only), at the cost of a resume having to supply fresh environment
            and carry states of its own. Anything that only loads weights
            (offline evaluation, inference) is unaffected either way.

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

        step_name = f"step_{step:010d}"
        step_dir = os.path.join(abs_directory, step_name)
        tmp_dir = os.path.join(abs_directory, f"{_TMP_PREFIX}{step_name}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # Split network state: everything except RngKey → orbax, RngKey → pickle.
        # orbax cannot handle JAX new-style PRNG key arrays.
        non_key_state, rng_key_state, _ = _split_net_state(training_state.networks)

        # The optimizer only contains float/int arrays; no key arrays.
        _, opt_state = nnx.split(training_state.optimizer)

        # Save parameter arrays with orbax. A fresh checkpointer is created per
        # call and immediately closed to ensure all async writes complete.
        checkpointer = ocp.StandardCheckpointer()
        try:
            checkpointer.save(os.path.join(tmp_dir, "networks"), non_key_state)
            checkpointer.save(os.path.join(tmp_dir, "optimizer"), opt_state)
        finally:
            checkpointer.close()

        # Save everything else with pickle (JAX arrays including PRNG keys are
        # pickle-safe).
        metadata = {
            "networks_rng_key_state": rng_key_state,
            "network_states": (
                training_state.network_states if include_env_state else None
            ),
            "env_states": training_state.env_states if include_env_state else None,
            "rng_key": training_state.rng_key,
            "steps_taken": training_state.steps_taken,
            "step": step,
            "config": config,
        }
        with open(os.path.join(tmp_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        # Publish atomically. Re-checkpointing a step that already exists (a
        # resumed run saving at the step it restored from) replaces it.
        if os.path.exists(step_dir):
            shutil.rmtree(step_dir)
        os.rename(tmp_dir, step_dir)

    return checkpoint_fn


def latest_checkpoint(directory: str) -> Optional[str]:
    """Path of the highest-numbered complete checkpoint in ``directory``.

    Returns None when ``directory`` does not exist or holds no checkpoint.
    Directories still being written are skipped: they are named
    ``.tmp-step_*`` until the save completes (see :func:`make_checkpoint_fn`).
    """
    if not os.path.isdir(directory):
        return None
    steps = []
    for name in os.listdir(directory):
        if not name.startswith("step_"):
            continue
        if not os.path.isdir(os.path.join(directory, name)):
            continue
        try:
            steps.append((int(name[len("step_"):]), name))
        except ValueError:
            continue
    if not steps:
        return None
    return os.path.join(directory, max(steps)[1])


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

    For a light checkpoint (written with ``include_env_state=False``) the
    returned ``training_state`` has ``env_states`` and ``network_states`` set to
    None; it cannot be passed to ``train_ppo(initial_state=...)`` as-is. Build
    fresh ones and splice them in, e.g.::

        state = dataclasses.replace(
            ckpt["training_state"],
            env_states=nnx.vmap(env.reset)(jax.random.split(key, n_envs)),
            network_states=networks.initialize_state(n_envs),
        )

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