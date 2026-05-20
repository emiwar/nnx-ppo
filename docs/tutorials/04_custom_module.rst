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

The four things to learn
------------------------

1. **Carry state vs. NNX state.**
2. **The ``context`` keyword.**
3. **NNX containers for variable-size children.**
4. **Where the four ``Context`` values come from.**

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
        Context,
        StatefulModule,
        StatefulModuleOutput,
    )


    class MovingAverage(StatefulModule):
        """Per-batch running mean over the last `k` inputs."""

        def __init__(self, feature_size: int, k: int):
            self.feature_size = feature_size
            self.k = k

        def __call__(self, state, x, *, context: Context = Context.INFERENCE):
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

The ``context`` keyword
-----------------------

Every :class:`StatefulModule` ``__call__`` takes a keyword-only
``context: Context`` argument. Containers thread it through; modules
read it when their behaviour depends on the lifecycle phase.

The four contexts:

- ``Context.ROLLOUT`` — collecting data by driving the environment.
- ``Context.LOSS_REPLAY`` — replaying the rollout to compute the PPO
  loss / gradients. Stored ``raw_action``\s are fed back to samplers
  so the replay reproduces rollout activations exactly.
- ``Context.INFERENCE`` — evaluation, distillation teacher, ad-hoc.
- ``Context.STATS_UPDATE`` — a second replay of the rollout, run
  after the gradient phase, whose only purpose is to fold the
  rollout's activations into running statistics.

For the full per-module behaviour table see
:doc:`../reference/contexts`.

The library is built on one rule about what ``__call__`` is allowed
to do to NNX variables:

    **No writes to NNX variables that affect the forward output,
    unless** ``context == Context.STATS_UPDATE``.

In other words: a forward pass is pure unless we are explicitly in
the stats-update phase. This makes the gradient-phase replay produce
the same activations as the rollout — the property the PPO loss
relies on. Pure read-only access to NNX variables is fine in any
context; pure scratch writes that do not affect the next forward
output are fine; writes that *change* the output of subsequent calls
are restricted.

Adding a context-aware running mean
-----------------------------------

Extend ``MovingAverage`` to also track a global running mean. It
should update only under ``STATS_UPDATE``:

.. code-block:: python

    class GlobalMeanStats(nnx.Variable):
        pass


    class MovingAverageWithStats(MovingAverage):
        def __init__(self, feature_size, k):
            super().__init__(feature_size, k)
            self.mean = GlobalMeanStats(jp.zeros(feature_size))
            self.count = GlobalMeanStats(jp.zeros((), jp.int32))

        def __call__(self, state, x, *, context: Context = Context.INFERENCE):
            if context == Context.STATS_UPDATE:
                # Welford-style update: incorporate this batch's mean.
                batch_count = x.shape[0]
                new_count = self.count[...] + batch_count
                delta = jp.mean(x, axis=0) - self.mean[...]
                self.mean[...] = self.mean[...] + delta * (batch_count / new_count)
                self.count[...] = new_count

            return super().__call__(state, x, context=context)

A few things to call out:

- :class:`nnx.Variable` (as opposed to :class:`nnx.Param`) is the
  right base for running statistics: NNX tracks it but it does not
  receive gradients.
- The ``if context == Context.STATS_UPDATE`` branch is the only
  place writes happen to ``self.mean`` / ``self.count``. Under any
  other context the layer is a pure forward.
- The training loop arranges for the rollout to be replayed once,
  after the gradient phase, with ``context=STATS_UPDATE`` and the
  stored ``raw_action``\s. So the values your stats branch sees are
  exactly the values the layer saw during the rollout.
  :class:`~nnx_ppo.networks.normalizer.Normalizer` uses this same
  pattern in production.

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

Where the four contexts come from
---------------------------------

You will almost never set ``context`` yourself. The library wires it
in at three places:

- :func:`~nnx_ppo.algorithms.rollout.unroll_env` defaults to
  ``Context.ROLLOUT`` for data collection;
- :func:`~nnx_ppo.algorithms.rollout.eval_rollout` uses
  ``Context.INFERENCE``;
- :func:`~nnx_ppo.algorithms.ppo.ppo_loss` threads
  ``Context.LOSS_REPLAY`` into the gradient phase;
- the post-gradient stats replay inside
  :func:`~nnx_ppo.algorithms.ppo.ppo_step` runs with
  ``Context.STATS_UPDATE``.

The default for ``__call__``'s ``context`` is
``Context.INFERENCE``, which is the conservative choice — no
implicit stats updates, no behaviour you did not ask for.

If you are calling your network outside of the training loop (a
distillation teacher, ad-hoc inference, a debugging script), pass
``context=Context.INFERENCE`` explicitly. If you want deterministic
actions in evaluation, that is a separate knob on the action sampler
(``nets.eval()`` flips it; see :doc:`../reference/ppo_adapter`).

Recap
-----

To write a :class:`StatefulModule`:

1. Decide what is carry state (per-batch, reset on episode boundary)
   and what is NNX state (per-network, persistent).
2. Implement ``__init__``, ``__call__(state, x, *, context=...)``,
   ``initialize_state(batch_size)``, and — if your carry state needs
   episode resets — ``reset_state(prev_state)``.
3. Read ``context`` only if your behaviour depends on the lifecycle
   phase. Respect the "no writes to NNX variables that affect
   forward output unless STATS_UPDATE" rule.
4. Wrap variable-size lists/dicts of sub-modules in
   :class:`nnx.List` / :class:`nnx.Dict`.

Your new module is now a drop-in :class:`StatefulModule` — composable
with :class:`Sequential`, embeddable as a population's ``compute``
or a connection's ``transform`` in a :class:`PopulationGraph`,
inspectable via :func:`nnx.iter_modules`, traceable through
``nnx.jit`` / ``nnx.grad``.
