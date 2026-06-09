# nnx-ppo

Experimental implementation of [Proximal Policy Optimization](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
in [JAX](https://github.com/google/jax), with first-class support for
recurrent/stateful networks. Networks are built with
[flax.nnx](https://flax.readthedocs.io); environments follow the
[MuJoCo Playground](https://playground.mujoco.org/) API.

> **Status:** experimental — the API may change without notice.

## Highlights

- **Stateful modules.** Recurrent layers, delayed connections, and noisy /
  variational populations are all first-class citizens. Carry state is
  threaded through rollout collection *and* multi-epoch loss replay, and is
  reset correctly when the environment resets — something other JAX RL
  libraries (e.g. Brax) do not natively support.
- **PyTree observations.** Observations can be arbitrary nested dicts, which
  makes it easy to route different streams (proprioception, vision,
  imitation targets, …) to different parts of a network.
- **PyTree actions and rewards.** Actions and rewards are also allowed to be
  PyTrees, which simplifies multi-actuator and multi-agent setups.

## Installation

```bash
pip install nnx-ppo
```

`nnx-ppo` installs the CPU build of JAX by default. For a CUDA 12 GPU build:

```bash
pip install nnx-ppo "jax[cuda12]"
```

Optional extras:

- `nnx_ppo[examples]` — `brax`, `wandb`, and
  [`playground`](https://pypi.org/project/playground/) (import name
  `mujoco_playground`) for the scripts in [examples/](examples/).
- `nnx_ppo[dev]` — test-suite dependencies (`pytest`, `pyright`, `beartype`,
  `absl-py`, plus `playground` for the env-driven tests).

## Quick example

```python
from flax import nnx
import mujoco_playground

from nnx_ppo.algorithms import ppo
from nnx_ppo.algorithms.config import TrainConfig, PPOConfig, EvalConfig
from nnx_ppo.networks.factories import make_mlp_actor_critic
from nnx_ppo.wrappers import episode_wrapper

env = mujoco_playground.registry.load("CartpoleBalance")
train_env = episode_wrapper.EpisodeWrapper(env, 1000)

rngs = nnx.Rngs(0)
nets = make_mlp_actor_critic(
    env.observation_size,
    env.action_size,
    actor_hidden_sizes=[64] * 4,
    critic_hidden_sizes=[256] * 2,
    rngs=rngs,
    normalize_obs=True,
)

result = ppo.train_ppo(
    train_env,
    nets,
    TrainConfig(
        ppo=PPOConfig(n_envs=1024, rollout_length=30, total_steps=10_000_000),
        eval=EvalConfig(enabled=True, every_steps=500_000, n_envs=64,
                        max_episode_length=1000),
    ),
    eval_env=env,
)
print(f"Final eval reward: {result.eval_history[-1]['episode_reward_mean']}")
```

See [examples/wandb_logging.py](examples/wandb_logging.py) for a complete
training script with W&B logging and video recording.

## Documentation

Full documentation, tutorials, and API reference are at
<https://nnx-ppo.readthedocs.io>.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
