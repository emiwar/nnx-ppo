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
``module(state, x, *, context=...) -> StatefulModuleOutput``. Containers
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
    from nnx_ppo.networks.containers import PPOActorCritic
    from nnx_ppo.networks.factories import make_mlp
    from nnx_ppo.networks.normalizer import Normalizer
    from nnx_ppo.algorithms.distributions import NormalTanhSampler

    rngs = nnx.Rngs(0)
    obs_size = 8       # example
    action_size = 2

    actor = make_mlp(
        [obs_size, 64, 64, 2 * action_size],
        rngs, activation=nnx.swish, activation_last_layer=False,
    )
    critic = make_mlp(
        [obs_size, 64, 64, 1],
        rngs, activation=nnx.swish, activation_last_layer=False,
    )
    sampler = NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=0.1)

    nets = PPOActorCritic(
        actor=actor,
        critic=critic,
        action_sampler=sampler,
        preprocessor=Normalizer(obs_size),
    )

:class:`~nnx_ppo.networks.containers.PPOActorCritic` is the convenience
wrapper for "one actor, one critic, one sampler, optional preprocessor".
Internally it builds a small graph that we are about to construct
explicitly when we need anything more flexible.

Note the actor's output size: ``2 * action_size``. The action sampler
expects ``[mean | log_std]`` concatenated along the last axis. For a
Gaussian-tanh policy the actor produces both; the sampler splits and
samples.

The containers
--------------

:class:`~nnx_ppo.networks.containers.Sequential`
    Chain layers. ``Sequential([a, b, c])`` runs ``a → b → c`` and
    routes each layer's carry state through a list keyed by position.

:class:`~nnx_ppo.networks.containers.Parallel`
    Run several named sub-modules on the **same** input. Returns a
    dict keyed by sub-module name. ``Parallel(action_params=actor,
    value=critic)`` produces ``{"action_params": ..., "value": ...}``
    from a single input.

:class:`~nnx_ppo.networks.containers.Concat`
    Pytree input → concatenate. Takes a dict ``{k: sub_input}``,
    runs each named sub-module on its own slice, and concatenates the
    outputs along the last axis. Used at the *start* of a stack when
    the observation is structured.

:class:`~nnx_ppo.networks.containers.Splitter`
    Inverse of :class:`Concat`. Takes a flat tensor and splits it into
    named slices: ``Splitter(action_params=2*A, value=1)`` turns a
    ``(B, 2A+1)`` tensor into ``{"action_params": (B, 2A), "value":
    (B, 1)}``. Used at the *end* of a stack to produce the dict
    output that an adapter consumes.

:class:`~nnx_ppo.networks.utils.Flattener`
    Pytree input → flat tensor (or dict-of-flat-tensors). With
    ``Flattener()`` every leaf is reshaped to ``(B, -1)`` and
    concatenated along the last axis. With
    ``Flattener(preserve_levels=N)`` the top ``N`` levels of
    ``dict`` / ``list`` / ``tuple`` structure are preserved and only
    the sub-trees below them are flattened — useful before a
    per-key consumer like :class:`Normalizer` or
    :class:`PopulationGraph` when each top-level value is itself a
    small pytree.

:class:`~nnx_ppo.networks.utils.Filter`
    Declarative pytree extraction / projection. Takes a dict
    ``{output_key: spec}``; each spec is a string (top-level key),
    a tuple of strings/ints (nested path), or a callable applied to
    the full input. Use it to ablate obs streams, give the critic
    privileged info, or reshape structured observations for a
    downstream consumer.

:class:`~nnx_ppo.networks.utils.Merge`
    Like :class:`Parallel`, but each sub-module must return a
    ``dict`` and :class:`Merge` flattens them into one combined
    dict (duplicate keys are an error). The natural complement when
    you have two independent stacks each emitting their own named
    heads, and the downstream consumer wants a single flat dict —
    for example a population graph that emits ``{motor_*}`` paired
    with a detached MLP critic that emits ``{value_*}``.

:class:`~nnx_ppo.networks.utils.Scale`
    Multiply by a fixed scalar. Use it on the output of a head
    when you want to scale action means / value estimates without
    baking the factor into a Dense initialiser.

The adapter
-----------

:class:`~nnx_ppo.algorithms.adapter.PPOAdapter` is the general-purpose
:class:`~nnx_ppo.networks.types.PPONetwork`. It wraps any
:class:`StatefulModule` whose forward output is a dict of named heads,
and it knows how to:

- run named action heads through declared samplers, and
- read named value heads into ``value_estimates``.

In the actor-critic shape, ``PPOActorCritic`` is literally the
following four-liner:

.. code-block:: python

    from nnx_ppo.algorithms.adapter import PPOAdapter
    from nnx_ppo.networks.containers import Sequential, Parallel

    inner = Sequential([
        Normalizer(obs_size),
        Parallel(action_params=actor, value=critic),
    ])
    nets = PPOAdapter(
        inner=inner,
        action_specs={"action_params": sampler},
        value_specs="value",
    )

If there is exactly one action head and one value head the adapter
unwraps the resulting single-key dicts back into bare arrays — so
``PPONetworkOutput.actions`` is an ``Array``, not ``{"action_params":
Array}``. With multiple heads it stays as a dict; you address each
head by its name. See :doc:`../reference/ppo_adapter` for the full
contract.

Worked example: encoder-decoder actor + shared-trunk critic
-----------------------------------------------------------

A more interesting case: split a Playground observation into a few
streams (say a ``proprio`` stream and a ``goal`` stream), encode them
separately, share a trunk, and emit an action and a value. We assume
the env exposes an obs dict::

    obs = {"proprio": (B, P), "goal": (B, G)}

Encode each stream with its own MLP, concatenate, then run a shared
trunk that produces action params and value through a final
:class:`Splitter`:

.. code-block:: python

    from nnx_ppo.networks.containers import Concat, Sequential, Splitter
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

    trunk = Sequential([
        # Per-stream encoders → concat along last axis.
        Concat(proprio=proprio_encoder, goal=goal_encoder),
        # Shared body.
        make_mlp([2 * H, H, H], rngs, activation=nnx.swish,
                 activation_last_layer=True),
        # Split the body's output into the action head and the value head.
        make_mlp([H, 2 * action_size + 1], rngs, activation=nnx.swish,
                 activation_last_layer=False),
        Splitter(action_params=2 * action_size, value=1),
    ])

    sampler = NormalTanhSampler(rngs, entropy_weight=1e-2, min_std=0.1)
    nets = PPOAdapter(
        inner=trunk,
        action_specs={"action_params": sampler},
        value_specs="value",
    )

What this network does, top-to-bottom:

1. The :class:`Concat` block expects the env's obs dict. It runs
   ``proprio_encoder`` on ``obs["proprio"]`` and ``goal_encoder`` on
   ``obs["goal"]``, then concatenates their outputs to a ``(B, 2H)``
   vector.
2. The next ``make_mlp`` is the shared body.
3. The third ``make_mlp`` produces the combined action / value head as
   a flat ``(B, 2A + 1)`` tensor.
4. :class:`Splitter` turns that into ``{"action_params": (B, 2A),
   "value": (B, 1)}``.
5. :class:`PPOAdapter` runs the sampler on ``action_params``,
   squeezes ``value``'s trailing axis, and packages everything into
   the standard :class:`~nnx_ppo.networks.types.PPONetworkOutput`.

Multi-head actor / critic
-------------------------

If your env produces multi-objective rewards or your policy has
multiple independent actions (one per body part, say), declare more
than one action sampler or value head:

.. code-block:: python

    nets = PPOAdapter(
        inner=trunk_emitting_arm_and_leg_heads,
        action_specs={
            "arm": NormalTanhSampler(rngs, entropy_weight=1e-2),
            "leg": NormalTanhSampler(rngs, entropy_weight=1e-2),
        },
        value_specs=["arm_value", "leg_value"],
    )

With multiple heads ``PPONetworkOutput.actions`` becomes ``{"arm":
..., "leg": ...}`` and ``value_estimates`` becomes
``{"arm_value": ..., "leg_value": ...}``. The PPO loss accepts either
shape — it computes GAE per reward key independently.

Privileged critic via ``Filter`` + ``Merge``
--------------------------------------------

Sometimes the critic should see strictly *more* than the actor — full
goal information, ground-truth distances, anything the actor must
infer at deployment. Express this with two parallel branches
operating on the same obs, each pulling out the subset they're
allowed to use, and :class:`~nnx_ppo.networks.utils.Merge` combining
their outputs:

.. code-block:: python

    from nnx_ppo.networks.utils import Filter, Merge

    actor_branch = Sequential([
        Filter({                                   # actor sees proprio only
            "proprio": ("proprio",),
        }),
        Flattener(),
        actor_mlp,
        Splitter(action_params=2 * action_size),
    ])
    critic_branch = Sequential([
        Filter({                                   # critic sees everything
            "proprio": ("proprio",),
            "goal":    ("goal",),
            "privileged_distance": ("goal", "dist"),
        }),
        Flattener(),
        critic_mlp,
        Splitter(value=1),
    ])

    inner = Sequential([
        Normalizer(obs_size),
        Merge(actor=actor_branch, critic=critic_branch),
    ])
    nets = PPOAdapter(
        inner=inner,
        action_specs={"action_params": sampler},
        value_specs="value",
    )

The two branches are completely independent: different obs slices,
different parameter sets, different depths. :class:`Merge` flattens
their dict outputs into ``{"action_params": ..., "value": ...}``
which feeds the adapter exactly as in the shared-trunk case.

What's next
-----------

Containers stop being expressive enough when your topology has
multiple connections feeding the same node, recurrent loops, or
per-connection delays. :doc:`03_graph` introduces
:class:`~nnx_ppo.networks.graph.PopulationGraph` for those cases.

If you need a layer the library does not provide, :doc:`04_custom_module`
walks through implementing :class:`StatefulModule` yourself.
