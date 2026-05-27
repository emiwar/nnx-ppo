Pytree utilities
================

The :mod:`nnx_ppo.networks.utils` module collects small stateless
:class:`~nnx_ppo.networks.types.StatefulModule` s for pytree projection,
reshaping, and per-key dispatch. They complement the four canonical
containers in :mod:`nnx_ppo.networks.containers`.

Flattener
---------

Pytree → flat tensor (or dict-of-flat-tensors).

.. code-block:: python

    from nnx_ppo.networks.utils import Flattener

    # depth-0: every leaf is reshaped to (B, -1) and concatenated.
    Flattener()
    # depth-1: preserves the top dict, flattens each value.
    Flattener(preserve_levels=1)

With the default ``preserve_levels=0`` every leaf is reshaped to
``(B, -1)`` and concatenated along the last axis, producing one flat
tensor. With ``preserve_levels=N``, the top ``N`` levels of ``dict`` /
``list`` / ``tuple`` structure are preserved and only the sub-trees
below them are flattened. So ``Flattener(preserve_levels=1)`` applied
to ``{"a": {"p": (B, 4), "t": (B, 8)}, "b": (B, 6)}`` returns
``{"a": (B, 12), "b": (B, 6)}``.

Idempotent on already-flat inputs at the appropriate depth: passing
``{"a": (B, 12), "b": (B, 6)}`` through ``Flattener(preserve_levels=1)``
is a no-op.

Filter
------

Declarative pytree extraction / projection. Output is a dict whose
keys are exactly the keys of the ``spec`` and whose values are pulled
from the input per spec entry.

.. code-block:: python

    from nnx_ppo.networks.utils import Filter

    Filter({
        "arm_L":  ("arm_L", "proprioception"),          # nested path
        "arm_R":  ("arm_R", "proprioception"),
        "head":   "head",                               # top-level key
        "joystick": ("root", "future_target", "pos"),
        "zero":   lambda obs: jp.zeros_like(obs["x"]),  # callable
    })

``spec`` is a dict ``{output_key: extraction}`` where each extraction
is:

- a **string** ``k`` — take ``x[k]``;
- a **tuple** of strings/ints ``(k1, k2, ...)`` — nested path,
  equivalent to ``x[k1][k2]...``;
- a **callable** ``fn`` — applied to the full input ``x``; ``fn(x)``
  becomes the value for ``output_key``.

Anything in the input not named is dropped. Use Filter to ablate obs
streams, give the critic privileged info, or reshape structured
observations before a downstream consumer.

Map
---

Per-key dispatch: dict input, dict output, with a different
sub-module per key.

.. code-block:: python

    from nnx_ppo.networks.utils import Map

    Map({pop: NormalTanhSampler(rngs, ...) for pop in POPULATIONS})

``Map({k: f for k in keys})`` applies each ``f`` to the upstream's
same-named entry. Extra keys in the upstream are dropped; missing
keys raise ``KeyError``.

Distinct from :class:`~nnx_ppo.networks.containers.Parallel` (same
input fed to every component) and
:class:`~nnx_ppo.networks.containers.Concat` (per-key dispatch with
concatenated output).

Carry state is a dict ``{name: component_state}``.

Merge
-----

Run named sub-modules on the **same** input; each must return a dict;
the dicts are merged into one flat dict.

.. code-block:: python

    from nnx_ppo.networks.utils import Merge

    inner = Merge(
        motors=graph,                       # emits {motor_arm_L, ...}
        critic=detached_critic_stack,       # emits {value_arm_L, ...}
    )

Duplicate keys across components are a hard error. The natural
complement to :class:`Parallel` when downstream consumers want one
flat dict of named heads but the heads come from independent
sub-networks.

Scale
-----

Multiply input by a fixed scalar factor.

.. code-block:: python

    from nnx_ppo.networks.utils import Scale

    Sequential([motor_head_dense, Scale(0.1)])  # damp motor output

Use this on the output of a head when you want to scale action means
or value estimates without baking the factor into the Dense
initialiser — keeps the factor visible to introspection / logging.

Construction forms
------------------

:class:`Map` and :class:`Merge` accept *either* keyword arguments *or*
a positional dict, e.g.::

    Map({pop: sampler for pop in POPULATIONS})

Use the positional form when keys aren't valid Python identifiers or
when you're building the container from a comprehension.
