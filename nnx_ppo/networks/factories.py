from typing import List, Callable, Union

from flax import nnx

from nnx_ppo.networks.containers import PPOActorCritic, Sequential
from nnx_ppo.networks.feedforward import Dense
from nnx_ppo.networks.sampling_layers import NormalTanhSampler
from nnx_ppo.networks.normalizer import Normalizer


def make_mlp_layers(
    sizes: List[int],
    rngs: nnx.Rngs,
    activation: Callable = nnx.relu,
    activation_last_layer: bool = True,
    **linear_kwargs
) -> List[Dense]:
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
    sizes: List[int],
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
    actor_hidden_sizes: List[int],
    critic_hidden_sizes: List[int],
    rngs: nnx.Rngs,
    activation: Union[Callable, str] = nnx.relu,
    normalize_obs: bool = True,
    initializer_scale: float = 1.0,
    # Sampler arguments
    entropy_weight: float = 1e-2,
    min_std: float = 1e-1,
    std_scale: float = 1.0,
) -> PPOActorCritic:
    """Create a simple MLP-based actor-critic network.

    Args:
        obs_size: Size of observation vector.
        action_size: Size of action vector.
        actor_hidden_sizes: List of hidden layer sizes for actor.
        critic_hidden_sizes: List of hidden layer sizes for critic.
        rngs: NNX random number generators.
        activation: Activation function.
        normalize_obs: Whether to normalize observations.
        initializer_scale: Scale for variance scaling initializer.
        entropy_weight: Entropy bonus weight for the action sampler.
        min_std: Minimum standard deviation for action distribution.
        std_scale: Scale factor for action standard deviation.

    Returns:
        A PPOActorCritic network.
    """
    if isinstance(activation, str):
        activation = {"swish": nnx.swish, "tanh": nnx.tanh, "relu": nnx.relu}[
            activation
        ]

    kernel_init = nnx.initializers.variance_scaling(
        initializer_scale, "fan_in", "uniform"
    )

    actor_sizes = [obs_size] + actor_hidden_sizes + [action_size * 2]
    actor = make_mlp(
        actor_sizes,
        rngs,
        activation,
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    critic_sizes = [obs_size] + critic_hidden_sizes + [1]
    critic = make_mlp(
        critic_sizes,
        rngs,
        activation,
        activation_last_layer=False,
        kernel_init=kernel_init,
    )

    action_sampler = NormalTanhSampler(
        rngs,
        entropy_weight=entropy_weight,
        min_std=min_std,
        std_scale=std_scale,
    )

    preprocessor = Normalizer(obs_size) if normalize_obs else None

    return PPOActorCritic(
        actor=actor,
        critic=critic,
        action_sampler=action_sampler,
        preprocessor=preprocessor,
    )
