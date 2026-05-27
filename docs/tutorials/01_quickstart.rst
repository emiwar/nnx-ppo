Quickstart
==========

This tutorial trains a plain MLP actor-critic on ``CartpoleSwingup``, a
small ``mujoco_playground`` environment.

Setup
-----

The tutorial assumes ``nnx-ppo``, ``flax``, ``jax``, and
``mujoco_playground`` are importable from the active environment.

.. code-block:: python

    import mujoco_playground
    from flax import nnx

    from nnx_ppo.algorithms import ppo
    from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig
    from nnx_ppo.networks.factories import make_mlp_actor_critic

Build the environment
---------------------

Pick any registered Playground env. ``CartpoleSwingup`` is small enough
to train in a few minutes on a laptop GPU.

.. code-block:: python

    env_name = "CartpoleSwingup"
    env = mujoco_playground.registry.load(env_name)

The full list of registered envs lives in :mod:`mujoco_playground.registry`.

Build the network
-----------------

:func:`~nnx_ppo.networks.factories.make_mlp_actor_critic` returns a
:class:`~nnx_ppo.networks.containers.Sequential` — an optional
observation normalizer followed by a
:class:`~nnx_ppo.networks.adapter.PPOAdapter` whose two ports own the
actor + sampler chain and the critic chain respectively.

.. code-block:: python

    SEED = 0
    rngs = nnx.Rngs(SEED)
    nets = make_mlp_actor_critic(
        obs_size=env.observation_size,
        action_size=env.action_size,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        rngs=rngs,
        activation=nnx.swish,
        normalize_obs=True,
    )

The factory creates two MLPs (actor and critic), an action sampler, and
— because ``normalize_obs=True`` — an observation normalizer that
running-standardises each observation dimension. ``rngs`` is an
:class:`flax.nnx.Rngs` keychain used for parameter initialisation and
for the stochastic action sampler. See
:doc:`02_composition` for how to build non-trivial networks by hand
instead of using the factory.

Note that ``activation`` and ``normalize_obs`` are **network-factory**
knobs (they shape how the network is built); they do not appear in the
:class:`PPOConfig` below, which only configures the training algorithm
that operates on whatever network you hand it.

Configure training
------------------

PPO runs in iterations. In each iteration, the current policy (actor
network) is first rolled out in the environment to collect a batch of
experience. The rollout is parallelised across multiple environments
to take advantage of GPU batching. Second, this batch of experience is
used as a dataset to update both the actor and the critic using
backpropagation and gradient descent.

The PPO algorithm has a number of hyperparameters that control this
process. In this library, these parameters are specified in a
:class:`~nnx_ppo.algorithms.config.PPOConfig`.

During training we typically want to monitor progress. We can do this
through regular evaluation runs (which are also batched across
multiple environments). Configuration for these evaluation runs is
specified in :class:`~nnx_ppo.algorithms.config.EvalConfig`.

Finally, these two configurations are combined into a
:class:`~nnx_ppo.algorithms.config.TrainConfig`, along with a seed.

.. code-block:: python

    config = TrainConfig(
        ppo=PPOConfig(
            n_envs=512,
            rollout_length=20,
            total_steps=2_000_000,
            n_epochs=4,
            n_minibatches=8,
            learning_rate=3e-4,
        ),
        eval=EvalConfig(
            enabled=True,
            every_steps=100_000,
            n_envs=64,
            max_episode_length=1000,
        ),
        seed=SEED,
    )

The :class:`PPOConfig` fields used above:

``n_envs``
    Number of environments rolled out in parallel per training
    iteration. Larger values give lower-variance gradient estimates at
    the cost of more GPU memory. The training loop applies
    :func:`jax.vmap` to the env across this batch for you — you do not
    need to wrap the env yourself.
``rollout_length``
    Number of environment steps collected per environment, per
    iteration. The total batch size per iteration is
    ``n_envs * rollout_length``.
``total_steps``
    Stop training once the total number of environment steps taken
    across all parallel envs reaches this. Rollout length and ``n_envs``
    are not adjusted — training simply ends after the iteration that
    pushes the cumulative step count past this threshold.
``n_epochs``
    Number of passes over each rollout batch during the gradient
    phase.
``n_minibatches``
    Number of minibatches each rollout batch is split into per epoch.
    Total gradient steps per iteration is ``n_epochs * n_minibatches``.
``learning_rate``
    Adam step size for the actor and critic.

Many more knobs (``clip_range``, ``gae_lambda``, ``discounting_factor``,
``entropy_weight``, weight decay, gradient clipping, …) are available
with sensible defaults. See :doc:`../reference/parameters` for the
full list with explanations.

The :class:`EvalConfig` fields used above:

``enabled``
    Whether to run periodic evaluation rollouts. With ``enabled=False``
    no evaluation is performed and the ``every_steps`` / ``n_envs`` /
    ``max_episode_length`` settings are ignored.
``every_steps``
    Approximate interval between eval runs, measured in cumulative
    environment steps.
``n_envs``
    Number of environments stepped in parallel during an eval rollout.
``max_episode_length``
    Each eval episode is cut off at this many steps. Useful for envs
    that don't terminate on their own.

The top-level ``seed`` controls JAX RNG initialisation throughout the
training loop. Use the same seed as the ``nnx.Rngs(SEED)`` you passed
to the network factory if you want fully reproducible runs.

Train
-----

:func:`~nnx_ppo.algorithms.ppo.train_ppo` runs the whole loop:
JIT-compiled rollouts, gradient updates, periodic eval, optional video
and checkpointing. A ``log_fn`` receives the per-iteration metrics
dict — useful for plotting eval curves without re-running eval
manually.

.. code-block:: python

    history = []

    def log_fn(metrics, steps):
        # train_ppo merges eval metrics into the per-iteration metrics
        # dict only on iterations where an eval ran. Use the presence of
        # episode_reward_mean as the "this step carried an eval" signal.
        if "episode_reward_mean" in metrics:
            history.append((steps, float(metrics["episode_reward_mean"])))
            print(f"step={steps}: reward={metrics['episode_reward_mean']:.2f}")

    result = ppo.train_ppo(env=env, networks=nets, config=config, log_fn=log_fn)

    print(f"Done: {result.total_steps} steps, "
          f"{result.total_iterations} PPO iterations")

``log_fn`` is called once per training iteration with
``(metrics: dict, steps: int)``. Metrics include the loss components
for that iteration; eval keys (``episode_reward_mean``,
``episode_reward_std``, …) are merged in only on iterations where eval
ran.

Inspecting the result
---------------------

:class:`~nnx_ppo.algorithms.config.TrainResult` exposes the final
:class:`~nnx_ppo.algorithms.types.TrainingState`, the eval history,
and the final metrics:

.. code-block:: python

    for entry in result.eval_history:
        print(entry["step"], entry.get("episode_reward_mean"))

The network itself is still ``nets`` — :func:`train_ppo` mutates it
in place. You can call ``nets.eval()`` to switch the action sampler
to deterministic (use the mean instead of sampling) and then run the
network forward yourself for ad-hoc inference.

To save and resume training across runs, pass a ``checkpoint_fn`` to
:func:`train_ppo` and reconstruct the matching network on the way
back — see :doc:`../reference/checkpointing` for the details.

Next steps
----------

If you need more complicated networks:

* :doc:`02_composition` — build encoder-decoder and multi-head networks
  from the standard containers.
* :doc:`03_graph` — populations and connections for modular or
  recurrent topologies.
* :doc:`04_custom_module` — implement your own :class:`StatefulModule`.
