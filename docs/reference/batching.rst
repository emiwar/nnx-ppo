Batching contract
=================

nnx-ppo's batching contract is uniform but stratified by interface:
:class:`StatefulModule` ``__call__`` runs over a flat batch with **no
time dimension**; the time scan is done outside, by the training and
rollout machinery. :meth:`update_statistics` is the one place where a
time dimension *does* surface. Environments are written for a single
env and are vmapped to ``n_envs`` automatically.

Networks: ``[B, *feat]`` per leaf
---------------------------------

Inside :meth:`StatefulModule.__call__`, every input leaf has shape
``(B, *feature_dims)``. There is no leading time axis — the scan over
``T`` rollout / replay steps happens *outside* the module, in
:func:`~nnx_ppo.algorithms.rollout.unroll_env` (during data collection)
and in the loss-replay scan inside
:func:`~nnx_ppo.algorithms.ppo.ppo_loss` (during the gradient phase).

Concretely, for an env with a flat ``(O,)`` observation and a
training config with ``n_envs=512``:

- ``obs`` inside a network ``__call__`` has shape ``(512, O)``.
- ``state["something"]`` has its leading axis equal to 512 too.
- The return arrays in :attr:`StatefulModuleOutput.output`,
  :attr:`next_state`, :attr:`rollout_extras`, etc. all carry the same
  leading axis.

For modules that need to aggregate *across* the batch (e.g. a
running-mean welford step would do ``jp.mean(x, axis=0)``), do that in
:meth:`update_statistics`, not in :meth:`__call__`. The forward path
should be pure per-sample — the algorithm and JAX transforms assume
it.

For dict-structured observations the same rule holds per leaf: an obs
of ``{"proprio": (B, P), "goal": (B, G)}`` has 2-D leaves with ``B``
as the leading axis.

``update_statistics``: ``[T, B, *feat]``
-----------------------------------------

:meth:`StatefulModule.update_statistics` is the **only** entry point
that sees a time dimension. It is called once per training step
*after* the gradient update, with the rollout's stacked
``rollout_extras`` history. Leaves of that history have shape
``(T, B, *feat)``, where ``T`` is the rollout length and ``B`` is the
number of envs.

A stats-bearing module typically reshapes the leading two axes
together and processes the result as a single batch::

    def update_statistics(self, rollout_extras):
        # rollout_extras leaves: (T, B, *feat)
        flat = jax.tree.map(
            lambda v: v.reshape((-1,) + v.shape[2:]), rollout_extras
        )
        # Now treat `flat` as a flat (T*B, *feat) batch.
        ...

If your module's emission is multi-dimensional or structured, the
same rule applies leaf-wise: drop the leading ``T`` into ``B`` and
operate on the combined axis.

Containers route ``update_statistics`` per child the same way they
route ``__call__``'s state and ``rollout_extras`` — the time axis
flows through transparently and only the leaves see it. See
:doc:`containers` for the custom-container contract.

Environments: one env, vmapped automatically
--------------------------------------------

Environments follow the `MuJoCo Playground
<https://playground.mujoco.org>`_ convention: ``env.reset(key)`` and
``env.step(state, action)`` are written for a **single** environment.
The training loop (and any user-side eval rollout helpers) vmaps the
env across ``n_envs`` automatically:

- :func:`~nnx_ppo.algorithms.rollout.unroll_env` wraps each
  ``env.step`` in :func:`jax.vmap`, and a fresh batch of reset keys
  via :func:`jax.random.split`.
- :func:`~nnx_ppo.algorithms.rollout.eval_rollout` does the same for
  evaluation rollouts.
- :func:`~nnx_ppo.algorithms.ppo.new_training_state` initialises the
  ``n_envs`` env states by ``nnx.vmap(env.reset)`` over independent
  RNG keys.

You never need to write a "vectorised" version of your env — the
single-env step / reset are enough. If your env's internal data has
fields without a batch dimension (e.g. shared MuJoCo model handles),
those are passed through unchanged; see
:func:`~nnx_ppo.algorithms.rollout.tree_where` for the per-env
selection on done-flag that the rollout uses to keep state shapes
consistent across the vmap.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Interface
     - Leading axes on each leaf
     - Who scans / vmaps
   * - :meth:`StatefulModule.__call__`
     - ``(B, *feat)``
     - The library's scan runs T steps outside ``__call__``
   * - :meth:`StatefulModule.update_statistics`
     - ``(T, B, *feat)``
     - The rollout scan stacked T per-step emissions
   * - ``env.reset(key)`` /
       ``env.step(state, action)``
     - unbatched single-env state
     - The training loop vmaps to ``n_envs`` automatically
