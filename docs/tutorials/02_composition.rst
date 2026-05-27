Custom networks 1: composing with containers
============================================

The :func:`~nnx_ppo.networks.factories.make_mlp_actor_critic` factory
from :doc:`01_quickstart` is enough when your network is one MLP for
the actor, one MLP for the critic, and an optional observation
normalizer in front. Beyond that — encoder-decoder actors, shared
trunks, structured observations — you assemble the network by hand
from container modules. This tutorial walks through that.

Mental model
------------

Every layer in ``nnx-ppo`` implements :class:`~nnx_ppo.networks.types.StatefulModule`.
A :class:`StatefulModule` is anything callable as
``module(state, x, rollout_extras=None) -> StatefulModuleOutput``. Containers
themselves are :class:`StatefulModule` s — they just compose children.

Two kinds of state are tracked separately:

- **NNX-tracked state** (params, RNG streams, running stats) lives on
  the module instance and is updated by NNX. It is not reset across
  episodes.
- **Carry state** is the ``state`` argument passed in and the
  ``next_state`` field of :class:`~nnx_ppo.networks.types.StatefulModuleOutput`.
  It is what gets reset when the env resets (LSTM hidden state,
  ``Delay`` buffer, RNG carry for variational layers, …).

The containers in :mod:`nnx_ppo.networks.containers` thread both kinds
of state to their children. You compose them; you do not write the
plumbing.

A hand-built MLP actor-critic
-----------------------------

To get a feel for the moving parts, rebuild what the quickstart
factory produces:

.. code-block:: python

    from flax import nnx
    from nnx_ppo.networks.adapter import PPOAdapter
    from nnx_ppo.networks.containers import Sequential
    from nnx_ppo.networks.feedforward import Dense
    from nnx_ppo.networks.normalizer import Normalizer
    from nnx_ppo.networks.sampling_layers import NormalTanhSampler

    rngs = nnx.Rngs(0)
    obs_size = 8       # example
    action_size = 2

    actor = Sequential([
        Dense(obs_size, 64, rngs, activation=nnx.swish),
        Dense(64, 64, rngs, activation=nnx.swish),
        Dense(64, 2*action_size, rngs),
        NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=0.1),
    ])
    critic = Sequential([
        Dense(obs_size, 64, rngs, activation=nnx.swish),
        Dense(64, 64, rngs, activation=nnx.swish),
        Dense(64, 1, rngs),
    ])
    nets = Sequential([
        Normalizer(obs_size),
        PPOAdapter(
            action=actor,
            value=critic,
        ),
    ])

This is exactly what :func:`~nnx_ppo.networks.factories.make_mlp_actor_critic`
produces. The :class:`PPOAdapter` is a regular layer in the
:class:`Sequential`; its two ports (``action=`` and ``value=``) both
receive the same input and run their own chains independently. Here,
we have placed a :class:`Normalizer` in sequence with the PPOAdapter,
so that the normalization becomes shared between the actor and critic.
In principle, we could also have given the actor and critic their own
normalizers, but that would have lead to doubling the total compute 
and memory required by normalization.

Finally, note the actor's output size: ``2 * action_size``. The action sampler
expects ``[mean | log_std]`` concatenated along the last axis. For a
Gaussian-tanh policy the actor produces both; the sampler splits and
samples.

The essential containers
------------------------

The worked examples below use a handful of containers. Each is a
:class:`StatefulModule` that composes others. Here are the ones you
will reach for first; the rest live in
:doc:`../reference/containers` and :doc:`../reference/utils`.

:class:`~nnx_ppo.networks.containers.Sequential`
    Chain layers. ``Sequential([a, b, c])`` runs ``a → b → c``.

:class:`~nnx_ppo.networks.containers.Concat`
    Per-stream encoder for structured (dict) observations. Runs each
    named sub-module on its own slice of the input dict and
    concatenates the outputs along the last axis.

:class:`~nnx_ppo.networks.utils.Flattener`
    Reshape a pytree to a single ``(B, D)`` tensor. With
    ``preserve_levels=N`` the top ``N`` levels of dict/list/tuple
    structure are preserved.

:class:`~nnx_ppo.networks.utils.Filter`
    Declarative pytree extraction / projection. Takes a dict
    ``{output_key: spec}`` where each spec is a string (top-level
    key), a tuple of strings/ints (nested path), or a callable
    applied to the full input. Use it to ablate obs streams or give
    the critic privileged info.

:class:`~nnx_ppo.networks.utils.Map`
    Per-key dispatch: dict input, dict output, with a different
    sub-module per key. ``Map({k: f for k in keys})`` applies each
    ``f`` to the upstream's same-named entry.

For Parallel, Splitter, Merge, Scale, and the full per-container
contracts (carry state shapes, construction forms, etc.) see the
reference pages.

The adapter
-----------

:class:`~nnx_ppo.networks.adapter.PPOAdapter` is the canonical leaf
that turns a network's forward output into a
:class:`~nnx_ppo.networks.types.PPONetworkOutput`. It is a regular
:class:`StatefulModule` and lives inside a :class:`Sequential` like
any other layer.

It has two ports — ``action`` and ``value`` — and both receive the
same upstream input. Each port runs its own chain. The action port
must emit a tree of *sampler dicts* (``{"action", "log_likelihood"}``
leaves, the standard
:class:`~nnx_ppo.networks.sampling_layers.ActionSampler` output);
the value port emits whatever value tensor your critic produces.

Both the ``action`` and ``value`` ports also accept (nested)
dictionaries of tensors  — see the multi-head example below.

See :doc:`../reference/ppo_adapter` for the full contract.

Worked example: encoder-decoder actor + shared-trunk critic
-----------------------------------------------------------

A more interesting case: split a Playground observation into a few
streams (say a ``proprio`` stream and a ``goal`` stream), encode them
separately, share a trunk, and emit an action and a value. We assume
the env exposes an obs dict::

    obs = {"proprio": (B, P), "goal": (B, G)}

Encode each stream with its own MLP, concatenate, then run a shared
trunk; the adapter's two ports each own a small head on top:

.. code-block:: python

    from nnx_ppo.networks.containers import Concat, Sequential
    from nnx_ppo.networks.factories import make_mlp

    rngs = nnx.Rngs(0)
    P, G, action_size = 12, 4, 3
    H = 64

    proprio_encoder = make_mlp(
        [P, H, H], rngs, activation=nnx.swish, activation_last_layer=True,
    )
    goal_encoder = make_mlp(
        [G, H, H], rngs, activation=nnx.swish, activation_last_layer=True,
    )

    shared_trunk = Sequential([
        # Per-stream encoders → concat along last axis.
        Concat(proprio=proprio_encoder, goal=goal_encoder),
        # Shared body.
        make_mlp([2 * H, H, H], rngs, activation=nnx.swish,
                 activation_last_layer=True),
    ])

    actor_head = make_mlp(
        [H, 2 * action_size], rngs, activation=nnx.swish,
        activation_last_layer=False,
    )
    critic_head = make_mlp(
        [H, 1], rngs, activation=nnx.swish, activation_last_layer=False,
    )
    sampler = NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=0.1)

    nets = Sequential([
        shared_trunk,
        PPOAdapter(
            action=Sequential([actor_head, sampler]),
            value=critic_head,
        ),
    ])

What this network does, top-to-bottom:

1. The :class:`Concat` block expects the env's obs dict. It runs
   ``proprio_encoder`` on ``obs["proprio"]`` and ``goal_encoder`` on
   ``obs["goal"]``, then concatenates their outputs to a ``(B, 2H)``
   vector.
2. The shared body MLP reduces it to ``(B, H)``.
3. The :class:`PPOAdapter`'s ``action`` port feeds the ``(B, H)``
   trunk output through ``actor_head`` (emits ``(B, 2A)``
   ``[mean | log_std]``) and then through the sampler (emits the
   sampler dict). The ``value`` port feeds the same ``(B, H)`` through
   ``critic_head`` (emits ``(B, 1)``).
4. The adapter walks the action port's output to pull out
   ``actions`` and ``loglikelihoods``, squeezes the value port's
   trailing axis, and packages everything into a
   :class:`~nnx_ppo.networks.types.PPONetworkOutput`.

Multi-head actor / critic
-------------------------

If your env produces multi-objective rewards or your policy has
multiple independent actions (one per body part, say), declare more
than one action sampler or value head:

.. code-block:: python

    from nnx_ppo.networks.utils import Filter, Map

    # Trunk emits {"action_arm": ..., "action_leg": ..., "value_arm": ..., "value_leg": ...}.
    nets = Sequential([
        shared_trunk_emitting_named_heads,
        PPOAdapter(
            action=Sequential([
                Filter({"arm": "action_arm", "leg": "action_leg"}),  # rename
                Map({                                                # per-key dispatch
                    "arm": NormalTanhSampler(rngs, entropy_weight=1e-2),
                    "leg": NormalTanhSampler(rngs, entropy_weight=1e-2),
                }),
            ]),
            value=Filter({"arm": "value_arm", "leg": "value_leg"}),
        ),
    ])

With multiple heads ``PPONetworkOutput.actions`` becomes
``{"arm": ..., "leg": ...}`` and ``value_estimates`` becomes
``{"arm": ..., "leg": ...}``. The PPO loss accepts either shape — it
computes GAE per reward key independently.

Privileged critic via per-port ``Filter``
-----------------------------------------

Sometimes the critic should see strictly *more* than the actor — full
goal information, ground-truth distances, anything the actor must
infer at deployment. Because the :class:`PPOAdapter`'s two ports are
already independent chains operating on the same upstream input, each
port can just :class:`Filter` the obs to the subset it's allowed to
use:

.. code-block:: python

    from nnx_ppo.networks.utils import Filter, Flattener

    actor_chain = Sequential([
        Filter({                                   # actor sees proprio only
            "proprio": ("proprio",),
        }),
        Flattener(),
        actor_mlp,
        sampler,
    ])
    critic_chain = Sequential([
        Filter({                                   # critic sees everything
            "proprio": ("proprio",),
            "goal":    ("goal",),
            "privileged_distance": ("goal", "dist"),
        }),
        Flattener(),
        critic_mlp,
    ])

    nets = Sequential([
        Normalizer(obs_size),
        PPOAdapter(action=actor_chain, value=critic_chain),
    ])

The two ports are completely independent: different obs slices,
different parameter sets, different depths. Each port owns its full
chain from upstream input through to its respective output shape.

What's next
-----------

Containers stop being expressive enough when your topology has
multiple connections feeding the same node, recurrent loops, or
per-connection delays. :doc:`03_graph` introduces
:class:`~nnx_ppo.networks.graph.PopulationGraph` for those cases.

If you need a layer the library does not provide, :doc:`04_custom_module`
walks through implementing :class:`StatefulModule` yourself.
