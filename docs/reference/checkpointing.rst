Checkpointing
=============

nnx-ppo can periodically write the full
:class:`~nnx_ppo.algorithms.types.TrainingState` to disk and reload it
later to resume training (or to load a trained policy for inference).
The checkpoint format is split across orbax (for the bulk array
state) and pickle (for new-style PRNG-key variables, which orbax does
not handle).

What's saved
------------

Each checkpoint persists every field of
:class:`~nnx_ppo.algorithms.types.TrainingState`:

- **Network params** (``nnx.Param`` arrays for the actor, critic,
  samplers, heads, and any custom submodules) — via orbax.
- **Non-param NNX variables** — e.g. :class:`Normalizer`'s running
  ``mean`` / ``M2`` / ``counter``, any custom ``nnx.Variable``
  subclasses your code defines — via orbax.
- **RNG variables** on the network (``nnx.RngKey``, ``nnx.RngCount``).
  The count goes through orbax; the key goes through pickle (because
  orbax does not yet support the ``key<…>`` dtype).
- **Optimizer state** — Adam moments, learning rate schedule state —
  via orbax.
- **Carry state** for the network and the env (``network_states``,
  ``env_states``).
- **The top-level training RNG key**, the cumulative step count, and
  optionally the :class:`~nnx_ppo.algorithms.config.TrainConfig` that
  produced the run.

Disk layout per checkpoint::

    {directory}/step_{step:010d}/
        networks/          # orbax: non-key network variables
        optimizer/         # orbax: optimizer state
        metadata.pkl       # RngKey vars + TrainingState fields + step + config

Saving during training
----------------------

Pass a ``checkpoint_fn`` to :func:`~nnx_ppo.algorithms.ppo.train_ppo`.
The :func:`~nnx_ppo.algorithms.checkpointing.make_checkpoint_fn`
helper builds one for you::

    from nnx_ppo.algorithms.ppo import train_ppo
    from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn

    result = train_ppo(
        env=env,
        networks=nets,
        config=config,
        checkpoint_fn=make_checkpoint_fn("/tmp/my_run", config=config),
    )

Checkpoints are written every
:attr:`~nnx_ppo.algorithms.config.TrainConfig.checkpoint_every_steps`
cumulative env steps. The ``config`` keyword is optional but
recommended: it stores the :class:`TrainConfig` alongside the
checkpoint so the run is self-describing.

A ``checkpoint_fn`` is just a callable
``(training_state: TrainingState, step: int) -> None`` — you can
plug in your own (e.g. to write to a remote object store) instead of
using :func:`make_checkpoint_fn`.

Resuming a run
--------------

Restoring a checkpoint requires a network instance with the *same
architecture* as the one that was saved. The network's params are
overwritten in-place by the checkpoint values; the architecture
itself is not reconstructed from disk.

::

    from nnx_ppo.algorithms import ppo
    from nnx_ppo.algorithms.checkpointing import load_checkpoint
    from nnx_ppo.networks.factories import make_mlp_actor_critic

    # 1. Rebuild the same network architecture you trained with.
    nets = make_mlp_actor_critic(...)  # same kwargs as the original run
    training_state = ppo.new_training_state(env, nets, n_envs, seed)

    # 2. Load: weights / optimizer / carry / RNGs are restored in place.
    ckpt = load_checkpoint(
        "/tmp/my_run/step_0000500000",
        training_state.networks,
        training_state.optimizer,
    )

    # 3. Continue training from where the checkpoint left off.
    result = ppo.train_ppo(
        env, nets, ckpt["config"],
        initial_state=ckpt["training_state"],
    )

The returned dict also contains ``ckpt["step"]`` (int) and
``ckpt["config"]`` (the persisted :class:`TrainConfig`, or ``None``
if none was stored).

Loading for inference only
--------------------------

To load a trained policy without resuming training, build a network
+ optimizer template, call :func:`load_checkpoint`, and then ignore
the optimizer / training-state fields::

    nets = make_mlp_actor_critic(...)
    training_state = ppo.new_training_state(env, nets, n_envs, seed)
    load_checkpoint("/tmp/my_run/step_0000500000",
                    training_state.networks, training_state.optimizer)
    nets.eval()       # deterministic action sampler
    # ... call nets(...) directly on observations from the env.
