Custom networks 3: writing your own :class:`StatefulModule`
===========================================================

The library ships a small but growing set of layers
(:class:`~nnx_ppo.networks.feedforward.Dense`,
:class:`~nnx_ppo.networks.recurrent.LSTM`,
:class:`~nnx_ppo.networks.normalizer.Normalizer`,
:class:`~nnx_ppo.networks.delay.Delay`,
:class:`~nnx_ppo.networks.variational.VariationalBottleneck`,
…). When you need something else, write your own
:class:`~nnx_ppo.networks.types.StatefulModule`. This tutorial walks
through the four things you need to know to do that.

What we'll build
----------------

A ``MovingAverage`` layer that maintains, for each batch element, a
running average of its last ``k`` inputs and emits that average. We
will also give it a running global mean across all observed inputs —
not because the layer needs it, but to show how a layer can update
NNX-tracked statistics safely.

The three things to learn
-------------------------

1. **Carry state vs. NNX state.**
2. **The ``rollout_extras`` argument and the ``update_statistics`` hook.**
3. **NNX containers for variable-size children.**

Carry state vs. NNX state
-------------------------

Every :class:`StatefulModule` has two kinds of state:

- **NNX-tracked state** lives on ``self`` as attributes typed
  :class:`nnx.Param`, :class:`nnx.Variable`, or sub-modules. NNX
  traces these through ``nnx.jit`` and ``nnx.grad`` automatically.
  Params get gradients; non-``Param`` variables (like running
  statistics) do not. NNX state is **not** reset on environment
  reset.

- **Carry state** is the explicit ``state`` argument to ``__call__``
  and the ``next_state`` field of the returned
  :class:`~nnx_ppo.networks.types.StatefulModuleOutput`. It carries
  per-batch quantities — LSTM hidden states, delay buffers, RNG
  carries for stochastic layers. The training loop **does** reset
  carry state when the environment resets via
  :meth:`~nnx_ppo.networks.types.StatefulModule.reset_state`.

Rule of thumb: if it's "per-batch transient state that should be
zeroed when an episode ends", it goes in carry state. If it's
"per-network state that survives episodes" (params, RNG streams,
running stats), it goes in NNX state.

For the moving average:

- The ring buffer of last-k inputs is carry state (each env's window
  resets when that env's episode resets).
- The running global mean / count are NNX state (a single
  per-network running statistic, not per-batch).

A first cut: forward pass and carry state
-----------------------------------------

.. code-block:: python

    import jax.numpy as jp
    from flax import nnx

    from nnx_ppo.networks.types import (
        StatefulModule,
        StatefulModuleOutput,
    )


    class MovingAverage(StatefulModule):
        """Per-batch running mean over the last `k` inputs."""

        def __init__(self, feature_size: int, k: int):
            self.feature_size = feature_size
            self.k = k

        def __call__(self, state, x, rollout_extras=None):
            # state["buffer"] : (B, k, feature_size)
            # state["idx"]    : (B,)  circular write pointer
            idx = state["idx"]
            batch = jp.arange(idx.shape[0])

            new_buffer = state["buffer"].at[batch, idx].set(x)
            new_idx = (idx + 1) % self.k

            mean = jp.mean(new_buffer, axis=1)
            return StatefulModuleOutput(
                next_state={"buffer": new_buffer, "idx": new_idx},
                output=mean,
                regularization_loss=jp.array(0.0),
                metrics={},
                rollout_extras=None,
            )

        def initialize_state(self, batch_size: int):
            return {
                "buffer": jp.zeros((batch_size, self.k, self.feature_size)),
                "idx": jp.zeros(batch_size, jp.int32),
            }

        def reset_state(self, prev_state):
            return {
                "buffer": jp.zeros_like(prev_state["buffer"]),
                "idx": jp.zeros_like(prev_state["idx"]),
            }

Some things to notice:

- ``__call__`` accepts and returns explicit carry state. It does
  **not** mutate ``self``. The caller is responsible for keeping the
  returned ``next_state`` alive for the next step.
- ``__init__`` only stores config (``feature_size``, ``k``). For real
  parameters you would assign :class:`nnx.Param` or sub-modules
  here.
- :meth:`reset_state` is what the training loop calls on
  per-environment episode boundaries (via
  :func:`~nnx_ppo.algorithms.rollout.unroll_env`). It must preserve
  shapes, since it runs under ``vmap``.

The ``rollout_extras`` argument
-------------------------------

Every :class:`StatefulModule` ``__call__`` takes a third positional
argument, ``rollout_extras``. It is a pytree shaped like ``state``,
threaded through every container. Modules that need to communicate
information from the rollout pass back to a later loss-replay pass
emit a value in the returned
:class:`StatefulModuleOutput.rollout_extras` and consume one via the
argument. The three phases:

- **Rollout** — caller passes ``rollout_extras=None``. Each module
  emits its replay snapshot (action samplers store the sampled raw
  action; normalizers store the input they just normalised). The
  rollout scan stacks these over T into
  :attr:`Transition.rollout_extras`.
- **Loss replay** — caller passes the stored slice back in. Modules
  that need replay info (notably action samplers) consume it to
  reproduce the actually-taken action's log-likelihood under the
  updated policy.
- **Inference** — caller passes nothing. Modules still emit but the
  caller drops the emission on the floor.

A module that needs to switch behaviour between "fresh sample" and
"use stored value" checks ``if rollout_extras is None``.

For the full per-module behaviour table see
:doc:`../reference/contexts`.

The library is built on one rule about what ``__call__`` is allowed
to do to NNX variables:

    **No writes to NNX variables that affect the forward output in
    ``__call__``, ever.**

Stats-bearing modules accumulate state by overriding
:meth:`update_statistics`, which is called once per training step
*after* the loss / gradient update, with the full rollout's
``[T, B, ...]`` slice of ``rollout_extras``. This keeps every
``__call__`` pure and the rollout / loss-replay numerically
identical.

Adding a running mean via ``update_statistics``
-----------------------------------------------

Extend ``MovingAverage`` to also track a global running mean. The
forward pass stays pure; the update is folded in by overriding
:meth:`update_statistics`:

.. code-block:: python

    class GlobalMeanStats(nnx.Variable):
        pass


    class MovingAverageWithStats(MovingAverage):
        def __init__(self, feature_size, k):
            super().__init__(feature_size, k)
            self.mean = GlobalMeanStats(jp.zeros(feature_size))
            self.count = GlobalMeanStats(jp.zeros((), jp.int32))

        def __call__(self, state, x, rollout_extras=None):
            # Forward stays pure; always emit `x` so update_statistics
            # can see it. Eval-only callers drop the emission on the floor.
            out = super().__call__(state, x)
            return out.replace(rollout_extras=x)

        def update_statistics(self, rollout_extras):
            # rollout_extras: [T, B, feature_size]
            flat = rollout_extras.reshape((-1,) + rollout_extras.shape[2:])
            batch_count = flat.shape[0]
            new_count = self.count[...] + batch_count
            delta = jp.mean(flat, axis=0) - self.mean[...]
            self.mean[...] = self.mean[...] + delta * (batch_count / new_count)
            self.count[...] = new_count

A few things to call out:

- :class:`nnx.Variable` (as opposed to :class:`nnx.Param`) is the
  right base for running statistics: NNX tracks it but it does not
  receive gradients.
- The :meth:`update_statistics` override is the *only* place writes
  happen to ``self.mean`` / ``self.count``. Forward stays pure.
- The training loop calls ``network.update_statistics(rollout.rollout_extras)``
  once per step, after the gradient update. Containers route per
  child. :class:`~nnx_ppo.networks.normalizer.Normalizer` uses this
  same pattern in production.

Containers for variable-size children
-------------------------------------

If your module holds a *fixed* set of sub-modules, you can just
assign them as attributes and NNX will trace them:

.. code-block:: python

    class TwoLayer(StatefulModule):
        def __init__(self, rngs):
            self.layer_a = Dense(8, 16, rngs)
            self.layer_b = Dense(16, 4, rngs)

If you hold a **list or dict** of sub-modules — particularly one
whose size depends on a constructor argument — wrap it in
:class:`nnx.List` or :class:`nnx.Dict`. A plain Python ``list`` /
``dict`` of modules will not be traced by NNX and your parameters
will silently disappear from the trainable set:

.. code-block:: python

    class StackOfLayers(StatefulModule):
        def __init__(self, sizes, rngs):
            self.layers = nnx.List([
                Dense(d_in, d_out, rngs)
                for d_in, d_out in zip(sizes[:-1], sizes[1:])
            ])

:class:`~nnx_ppo.networks.containers.Sequential` is itself written
exactly this way — it stores its layers in an :class:`nnx.List`.

Where ``rollout_extras`` comes from
-----------------------------------

You will almost never pass ``rollout_extras`` by hand. The library
wires it in at three places:

- :func:`~nnx_ppo.algorithms.rollout.unroll_env` (data collection)
  calls the network with ``rollout_extras=None`` and stores each
  module's emission into ``Transition.rollout_extras``;
- :func:`~nnx_ppo.algorithms.rollout.eval_rollout` (evaluation) also
  passes ``rollout_extras=None`` and discards the emissions;
- :func:`~nnx_ppo.algorithms.ppo.ppo_loss` (the gradient phase) feeds
  each per-step slice of the stored extras back into the network.

If you want deterministic actions in evaluation, that is a separate
knob on the action sampler (``nets.eval()`` flips it; see
:doc:`../reference/ppo_adapter`).

Recap
-----

To write a :class:`StatefulModule`:

1. Decide what is carry state (per-batch, reset on episode boundary)
   and what is NNX state (per-network, persistent).
2. Implement ``__init__``, ``__call__(state, x, rollout_extras=None)``,
   ``initialize_state(batch_size)``, and — if your carry state needs
   episode resets — ``reset_state(prev_state)``.
3. Read ``rollout_extras`` only if your behaviour depends on it
   (consume the stored value when non-``None``; sample fresh when
   ``None``). Respect the "no writes to NNX variables that affect
   forward output in ``__call__``, ever" rule; defer accumulation to
   :meth:`update_statistics`.
4. Wrap variable-size lists/dicts of sub-modules in
   :class:`nnx.List` / :class:`nnx.Dict`.

Your new module is now a drop-in :class:`StatefulModule` — composable
with :class:`Sequential`, embeddable as a population's ``compute``
or a connection's ``transform`` in a :class:`PopulationGraph`,
inspectable via :func:`nnx.iter_modules`, traceable through
``nnx.jit`` / ``nnx.grad``.
