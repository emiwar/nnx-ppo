import jax
import jax.numpy as jp
from flax import nnx

class MyModule(nnx.Module):

  def __init__(self, rngs):
    self.rngs = rngs

  def __call__(self, x):
    return x + jax.random.normal(self.rngs())

SEED = 17
m = MyModule(nnx.Rngs(SEED))
nnx.jit(m)(0)
#scan_transform = nnx.scan(m, length=5, in_axes=nnx.Carry, out_axes=nnx.Carry)
#scan_transform(0.0)

@nnx.jit
def call_m(m, x):
  return m(x)

nnx.vmap(call_m, in_axes=(None, 0))(m, jp.zeros(5))


@nnx.jit
def call_m_carry(m, x):
  return x, m(x)

w_split_rngs = nnx.split_rngs(splits=10)(call_m_carry)

nnx.scan(call_m_carry, in_axes=(nnx.StateAxes({nnx.RngState: nnx.Carry}), nnx.Carry), out_axes=(nnx.Carry, 0), length=5)(m, 0.0)