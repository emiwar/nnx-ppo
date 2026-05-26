from typing import Union
from collections.abc import Callable

from flax import nnx

from nnx_ppo.networks.adapter import PPOAdapter
from nnx_ppo.networks.containers import Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.normalizer import Normalizer
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.types import StatefulModule


def make_mlp_layers(
    sizes: list[int],
    rngs: nnx.Rngs,
    activation: Callable = nnx.relu,
    activation_last_layer: bool = True,
    **linear_kwargs
) -> list[Dense]:
    """Create a list of Dense layers for an MLP.

    Use this when embedding MLP layers within another Sequential:
        Sequential([
            Flattener(),
            *make_mlp_layers([64, 32, 16], rngs),
            SomeOtherModule(),
        ])

    Args:
        sizes: List of layer sizes including input and output.
        rngs: NNX random number generators.
        activation: Activation function to use between layers.
        activation_last_layer: Whether to apply activation after the last layer.
        **linear_kwargs: Additional arguments passed to nnx.Linear.

    Returns:
        A list of Dense layers.
    """
    layers = []
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        is_last = i == len(sizes) - 2
        act = activation if (not is_last or activation_last_layer) else None
        layers.append(Dense(din, dout, rngs, activation=act, **linear_kwargs))
    return layers


def make_mlp(
    sizes: list[int],
    rngs: nnx.Rngs,
    activation: Callable = nnx.relu,
    activation_last_layer: bool = True,
    **linear_kwargs
) -> Sequential:
    """Create an MLP as a Sequential of Dense layers.

    Args:
        sizes: List of layer sizes including input and output.
        rngs: NNX random number generators.
        activation: Activation function to use between layers.
        activation_last_layer: Whether to apply activation after the last layer.
        **linear_kwargs: Additional arguments passed to nnx.Linear.

    Returns:
        A Sequential container of Dense layers.
    """
    return Sequential(
        make_mlp_layers(sizes, rngs, activation, activation_last_layer, **linear_kwargs)
    )


def make_mlp_actor_critic(
    obs_size: int,
    action_size: int,
    actor_hidden_sizes: list[int],
    critic_hidden_sizes: list[int],
    rngs: nnx.Rngs,
    activation: Union[Callable, str] = nnx.relu,
    normalize_obs: bool = True,
    initializer_scale: float = 1.0,
    # Sampler arguments
    entropy_weight: float = 1e-2,
    min_std: float = 1e-1,
    std_scale: float = 1.0,
) -> StatefulModule:
    """Build a standard one-actor / one-critic PPO network.

    Returns a ``Sequential`` whose forward output is a
    :class:`~nnx_ppo.networks.types.PPONetworkOutput`. Pass it straight to
    :func:`~nnx_ppo.algorithms.ppo.train_ppo`.

    The constructed pipeline is::

        Sequential([
            Normalizer(obs_size)?,        # if normalize_obs
            PPOAdapter(
                action=Sequential([actor, NormalTanhSampler(...)]),
                value=critic,
            ),
        ])

    Both adapter ports receive the same upstream input (the normalised obs),
    so there is no shared trunk; the actor and critic each run independently.
    Insert a shared trunk by prepending it to the Sequential and pointing the
    ports at it.
    """
    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    actor_layers = make_mlp_layers(
        [obs_size] + actor_hidden_sizes + [action_size * 2],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    critic = make_mlp(
        [obs_size] + critic_hidden_sizes + [1],
        rngs,
        activation,  # type: ignore[arg-type]
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    sampler = NormalTanhSampler(
        rngs,
        entropy_weight=entropy_weight,
        min_std=min_std,
        std_scale=std_scale,
    )

    # Sampler is just the last layer of the actor chain.
    adapter = PPOAdapter(
        action=Sequential([*actor_layers, sampler]),
        value=critic,
    )
    if normalize_obs:
        return Sequential([Normalizer(obs_size), adapter])
    return adapter
