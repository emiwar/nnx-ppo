from flax import nnx
from jax.experimental import checkify

def f(x):
    checkify.check(x>0, "x must be positive")
    return 2*x


f_jit = nnx.jit(f)
err, val = checkify.checkify(f_jit)(-17)
err.throw()