:class:`PPOAdapter` reference
=============================

:class:`~nnx_ppo.networks.adapter.PPOAdapter` is the canonical leaf
that turns a network's forward output into a
:class:`~nnx_ppo.networks.types.PPONetworkOutput`. It is a regular
:class:`~nnx_ppo.networks.types.StatefulModule` that lives inside a
``Sequential`` (or any other container) like any other layer. Its
forward output's ``output`` field is the ``PPONetworkOutput``.

Two-port router
---------------

``PPOAdapter`` has two ports — ``action`` and ``value`` — and both
receive the **same upstream input**. The action port emits a tree of
*sampler dicts* (each ``{"action", "log_likelihood"}``); the value
port emits whatever value-shape your critic produces. The adapter
walks the action port's output to assemble ``PPONetworkOutput.actions``
and ``loglikelihoods``, takes the value port's output as
``value_estimates`` (trailing singleton axis squeezed), and packages
everything.

.. code-block:: python

    PPOAdapter(action: StatefulModule, value: StatefulModule)

Minimal one-actor / one-critic
------------------------------

Each port owns its full chain. Both run on the same upstream tensor.

.. code-block:: python

    network = Sequential([
        Normalizer(obs_size),
        PPOAdapter(
            action=Sequential([actor_mlp, NormalTanhSampler(rngs, entropy_weight=1e-2)]),
            value=critic_mlp,
        ),
    ])

The factory :func:`~nnx_ppo.networks.factories.make_mlp_actor_critic`
builds exactly this shape.

Multi-head with per-key dispatch
--------------------------------

When the trunk emits a dict of named action-params and named values,
the adapter ports use :class:`~nnx_ppo.networks.utils.Filter` (to
extract / rename) and :class:`~nnx_ppo.networks.utils.Map` (to
dispatch per key):

.. code-block:: python

    network = Sequential([
        Normalizer(obs_size),
        graph,                                   # emits {"action_a": ..., "action_b": ...,
                                                 #         "value_a": ...,  "value_b": ...}
        PPOAdapter(
            action=Sequential([
                Filter({"a": "action_a", "b": "action_b"}),   # rename
                Map({"a": sampler_a, "b": sampler_b}),         # per-key dispatch
            ]),
            value=Filter({"a": "value_a", "b": "value_b"}),
        ),
    ])

``out.actions`` is then ``{"a": ..., "b": ...}``, matching the env's
per-key action spec.

Shared trunk
------------

To share computation between the action and value pathways, prepend
the trunk to the outer ``Sequential`` and let both ports consume its
output:

.. code-block:: python

    Sequential([
        Normalizer(obs_size),
        shared_trunk,                            # heavy computation
        PPOAdapter(
            action=Sequential([actor_head, sampler]),
            value=critic_head,
        ),
    ])

Action port output shape
------------------------

The action port's output should be a tree of *sampler dicts*. A
sampler dict is a leaf-position
``{"action": ..., "log_likelihood": ...}`` (the standard shape that
:class:`~nnx_ppo.networks.sampling_layers.ActionSampler` emits). The
adapter walks the tree with ``jax.tree.map(..., is_leaf=...)`` so:

- A bare sampler dict → ``out.actions`` is a bare array,
  ``out.loglikelihoods`` is a bare array.
- ``Map({k: sampler})`` → ``{k: sampler_dict}`` →
  ``out.actions = {k: array}``, ``out.loglikelihoods = {k: array}``.

Value port output shape
-----------------------

The value port's output is used as-is for
``PPONetworkOutput.value_estimates``, with each leaf squeezed if it
has a trailing length-1 axis. So a critic emitting ``(B, 1)`` lands
as a ``(B,)`` value estimate; a dict of critics emitting
``{k: (B, 1)}`` lands as ``{k: (B,)}``.

Carry state
-----------

The adapter's carry state is::

    {"action": <action port's carry>,
     "value":  <value port's carry>}

``initialize_state`` and ``reset_state`` route per port.

``rollout_extras`` and ``update_statistics``
--------------------------------------------

The adapter routes both ports' ``rollout_extras`` analogously to
state: ``rollout_extras["action"]`` to the action port,
``rollout_extras["value"]`` to the value port. The same routing is
used by :meth:`update_statistics`.

``train()`` / ``eval()``
------------------------

:meth:`nnx.Module.eval` and :meth:`nnx.Module.train` set the
``deterministic`` attribute recursively across the module tree. The
action sampler reads it to decide between sampling and returning the
mean. Conventional usage:

- Default: ``network.train()`` — samples stochastically (this is what
  the training loop does between rollouts).
- For deterministic eval / video / deployment: ``network.eval()``
  before the inference call; the PPO training loop already does this
  bookend around eval rollouts and restores ``train()`` afterwards.

During loss replay, the stored raw action from ``rollout_extras`` is
used and the sampler's fresh draw is discarded, so ``train()`` /
``eval()`` only affect calls that pass ``rollout_extras=None``.
