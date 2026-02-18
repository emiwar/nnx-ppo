import jax
import jax.numpy as jp
from vnl_playground.tasks.rodent.imitation import Imitation

env = Imitation(config_overrides={"solver": "newton"})
SEED = 45
env_state = jax.jit(env.reset)(jax.random.key(SEED))
step_env = jax.jit(env.step)

for i in range(50):
    print(f"Iter {i}")
    cache_size = step_env._cache_size()
    print(f"\t step_env cache size: {cache_size}")
    actions = jp.tanh(jax.random.normal(jax.random.key(SEED+i+1), (env.action_size,)))
    env_state = step_env(env_state, actions)
    if env_state.done:
        print("\tEnv is done.")
        for k,v in env_state.metrics.items():
            if k.startswith("termination"):
                print(f"\t{k}: {v}")