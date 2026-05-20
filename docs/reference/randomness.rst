Randomness
==========

Any layer that does stochastic sampling (a Gaussian sample, a noisy
neuron, a stochastic dropout mask, a Bernoulli gate) needs an RNG
key. **Where you put that key matters for gradient correctness.**
This page sets out the rule and explains the two reasons behind it.

The rule
--------

For a stochastic :class:`~nnx_ppo.networks.types.StatefulModule`, the
RNG key lives in the **carry state** — returned from
:meth:`initialize_state`, threaded through :meth:`__call__`, refreshed
by :meth:`reset_state` if appropriate. Specifically: keep one key
**per env**, of shape ``[B]``, advanced independently per env on each
forward step.

Do **not** keep the key as a class-level :class:`nnx.Variable` (or
``nnx.RngKey``) that you read and advance inside :meth:`__call__`.
That looks natural — it's how stateful modules typically work in
non-RL contexts — but it produces silently wrong gradients in
nnx-ppo.

Two reasons not to use a class-level RNG
----------------------------------------

The rollout / loss-replay split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each training iteration runs the network twice:

- In ``Context.ROLLOUT`` to drive the environment and collect data.
- Then in ``Context.LOSS_REPLAY`` to recompute activations on the
  recorded observations so the PPO loss can be differentiated.

The library is built on a guarantee that the second pass reproduces
exactly the activations of the first (see :doc:`contexts` for the
write rule that enforces this). Stored ``raw_action``\s feed the
samplers; deterministic forward operations behave identically.

If you advance an ``nnx.Variable`` RNG inside :meth:`__call__`, the
rollout pass will leave that variable in its post-rollout state. The
loss replay then starts from the *post-rollout* RNG and produces
different samples than the rollout did. PPO's importance ratios are
computed against the rollout activations, so the gradient now points
in a meaningless direction.

The minibatching caveat
~~~~~~~~~~~~~~~~~~~~~~~

You might think: "fine, I'll snapshot the RNG at the start of the
rollout and restore it before the loss replay." That fixes the
split, but it does not fix a deeper problem.

The rollout calls the network once per timestep on the full
``[n_envs, ...]`` batch. The loss replay, by contrast, runs one
**minibatch** at a time with batch size ``n_envs / n_minibatches``.
Even with the same starting key, ``jax.random.split`` called on a
single-stream key advances differently when the per-call batch shape
changes — the rollout and the replay produce different *sequences*
of per-call subkeys, so they disagree on which random number each
env sees.

There is no single-stream RNG carried on the module that can
reconcile both call patterns. The fix has to make the per-env RNG
state structurally independent of how the batch is sliced.

The pattern: one RNG per env, in carry state
---------------------------------------------

Carry an RNG **per env** as part of the module's carry state. Because
the per-env carry is sliced by JAX along with every other batched
quantity (rewards, dones, obs, the network's other carry state), the
per-env RNG stays in sync with the env it belongs to across rollout,
minibatching, and replay alike. Splitting that RNG inside
:meth:`__call__` is a pure local operation on the per-env carry — it
doesn't depend on the global batch shape.

:class:`~nnx_ppo.networks.variational.VariationalBottleneck` is the
canonical worked example::

    class VariationalBottleneck(StatefulModule):
        def __init__(self, latent_size, rng, kl_weight, min_std=1e-6):
            self.rng = rng                       # nnx.Rngs — used only at init
            ...

        def initialize_state(self, batch_size):
            # Build the per-env carry: B independent keys, derived once
            # at construction time from the module's class-level RNG.
            return jax.random.split(self.rng(), batch_size)

        def __call__(self, key, x, *, context=Context.INFERENCE):
            # `key` has shape [B]. Each env's key is split locally; the
            # new keys become the next carry. No class-level RNG is read
            # inside __call__.
            eps = jax.vmap(lambda k: jax.random.normal(k, (self.latent_size,)))(key)
            ...
            next_key, _ = jax.vmap(jax.random.split, out_axes=1)(key)
            return StatefulModuleOutput(next_state=next_key, output=z, ...)

The class-level :class:`nnx.Rngs` is consulted **once**, in
:meth:`initialize_state`, to seed the per-env carry. After that, the
forward pass is pure on its inputs — including the carry RNG keys —
and the rollout/replay/minibatching machinery handles the rest for
free.

If your env-reset semantics call for a fresh RNG on reset, implement
:meth:`reset_state` to re-split the class-level RNG. If you want the
key chain to survive resets (which is what
:class:`VariationalBottleneck` does), have :meth:`reset_state` return
``prev_state`` unchanged. Either is fine — what matters is that
within an episode, the per-env carry advances deterministically from
its initial value.

When you can ignore this
------------------------

If your module is not stochastic — :class:`Dense`, :class:`LSTM`,
:class:`Concat`, anything that does no sampling inside
:meth:`__call__` — there is no RNG to manage and nothing to do.

If your module's only randomness is in **parameter initialisation**
(read once at construction time), pass an :class:`nnx.Rngs` keychain
to ``__init__`` and use it there. That falls outside the rule: init
isn't called in the forward path.

The :class:`~nnx_ppo.algorithms.distributions.ActionSampler` family
is the one place where module-internal RNG advancement is OK —
because the loss-replay pass receives the stored ``raw_action`` and
**ignores** the freshly-sampled value. Even there, the RNG is
advanced consistently across contexts so downstream stochastic
layers stay in lockstep.
