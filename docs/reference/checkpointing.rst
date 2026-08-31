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

Each checkpoint is assembled in a ``.tmp-step_…`` directory and renamed
into place once complete, so a process killed mid-write leaves nothing
that looks like a finished checkpoint.
:func:`~nnx_ppo.algorithms.checkpointing.latest_checkpoint` returns the
newest complete one (and skips the temporaries).

Light checkpoints
-----------------

For an environment with a large per-env state, ``env_states`` can be
almost all of a checkpoint's size and write time. Pass
``include_env_state=False`` to leave ``env_states`` and
``network_states`` out::

    make_checkpoint_fn("/tmp/my_run", config=config, include_env_state=False)

What remains — params, non-param variables, optimizer state, RNGs and
``steps_taken`` — is small and quick to write, which makes frequent
checkpointing cheap and a save under a deadline safe. The trade is that
resuming needs fresh env and carry states of its own (see below);
anything that only loads weights is unaffected.

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

Resuming from a light checkpoint
--------------------------------

A light checkpoint (``include_env_state=False``) restores with
``env_states`` and ``network_states`` set to ``None``, so supply your
own before passing it to :func:`train_ppo`::

    import dataclasses, jax
    from flax import nnx

    ckpt = load_checkpoint(step_dir, template.networks, template.optimizer)
    resumed = dataclasses.replace(
        ckpt["training_state"],
        env_states=nnx.vmap(env.reset)(jax.random.split(key, n_envs)),
        network_states=nets.initialize_state(n_envs),
    )

Because the number of envs is no longer baked into the checkpoint, this
is also how you resume at a different ``n_envs``. Beware that resetting
every env at once *synchronises* them: if episodes have a similar
length, they then all begin and end together, which is not the
distribution steady-state training sees. Where that matters, reset each
env at a randomly chosen point in the episode instead of at the start.

Resuming an interrupted run
---------------------------

Two more arguments let a run be stopped and restarted, e.g. under a job
scheduler that may kill it at any time:

- ``stop_fn(steps) -> bool`` is called once per iteration; when it
  returns True the loop writes a checkpoint (unless one was just
  written at that step) and returns. Point it at a flag set by a
  ``SIGTERM`` handler and an interrupted run saves at the next iteration
  boundary instead of losing everything since the last scheduled
  checkpoint. To tell a stop from a completion, compare
  ``TrainResult.total_steps`` against the total you asked for: the loop
  exits on its own only once ``steps_taken`` has reached it.
- ``initial_eval=False`` skips the eval, video and checkpoint that
  otherwise run before the first iteration. Pass it when resuming: that
  work was already done at the step being restored, and it also keeps
  the interval trackers from firing again immediately after it.

::

    watcher = ...   # sets .triggered on SIGTERM
    result = ppo.train_ppo(
        env, nets, config,
        initial_state=resumed,
        checkpoint_fn=make_checkpoint_fn(run_dir, config,
                                         include_env_state=False),
        stop_fn=lambda steps: watcher.triggered,
        initial_eval=False,
    )
    interrupted = result.total_steps < config.ppo.total_steps

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
