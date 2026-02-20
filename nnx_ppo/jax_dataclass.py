"""JaxDataclass: a frozen dataclass mixin that registers as a JAX pytree.

Motivation: flax.struct.dataclass registers pytrees using __init__ in tree_unflatten.
When beartype/jaxtyping are active (via install_import_hook), they instrument __init__
with runtime type checks. During JAX pytree reconstruction inside jit/vmap/scan, JAX
may pass None sentinels or abstract tracer values, which fail the type checks.

This mixin uses object.__new__ in tree_unflatten, bypassing __init__ entirely during
reconstruction, so beartype never sees the intermediate values.

See: https://github.com/patrick-kidger/jaxtyping/issues/177
"""

import dataclasses


class JaxDataclass:
    """Frozen dataclass mixin that registers as a JAX pytree.

    Usage:
        @jax.tree_util.register_pytree_node_class
        @dataclasses.dataclass(frozen=True)
        class MyStruct(JaxDataclass):
            x: Float[Array, "batch"]
            y: int
    """

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def tree_flatten(self):
        fields = dataclasses.fields(self)
        return [getattr(self, f.name) for f in fields], [f.name for f in fields]

    @classmethod
    def tree_unflatten(cls, names, children):
        # Bypass __init__ so beartype does not check during JAX pytree reconstruction
        obj = object.__new__(cls)
        for name, child in zip(names, children):
            object.__setattr__(obj, name, child)
        return obj
