nnx-ppo
=======

This library is an experimental implementation of
`Proximal Policy Optimization <https://en.wikipedia.org/wiki/Proximal_policy_optimization>`_
(PPO) in `JAX <https://github.com/google/jax>`_, built on top of
`Flax NNX <https://flax.readthedocs.io>`_.

It is built with the aim of exploring neural network architectures inspired
by, or relevant to, neuroscience. As such, the aim is to enable training of
unconventional architectures, or architectures with various constraints.
This is supported through a set of features:

* First-class support for stateful modules. This enables recurrent networks,
  delayed connections, and other useful constructs. Additionally, by treating
  the RNG key as state, we get automatic support for variational layers and
  noisy populations.

  Stateful modules require care to integrate with reinforcement learning.
  For example, for consistency the network state should reset when the RL
  environment resets, and network state must be correctly handled through
  rollout collection and multiple gradient update batches on the collected
  experience. This is not natively supported by other JAX RL libraries such
  as `Brax <https://github.com/google/brax>`_.

* Support for observations as general PyTrees rather than plain tensors.
  This greatly simplifies routing components of the environment observations
  to specific network modules. Such routing is helpful in imitation tasks
  where we might want to route the proprioception and imitation target to
  different parts of the network (as in the encoder-decoder architecture in
  `MIMIC-MJX <https://mimic-mjx.talmolab.org/>`_), or to route vision input
  into a convolutional module, or to route observations from different
  bodies of a plant to different parts of a graph network.

* Support for actions and rewards as dictionaries / PyTrees. In addition to
  simplifying routing from the network output to the environment, dictionary
  actions and rewards provide the foundation for full multi-agent RL setups.

Status
------

nnx-ppo is **experimental** — the API may change without notice.

Start with :doc:`tutorials/01_quickstart` if you just want to start training.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/01_quickstart
   tutorials/02_composition
   tutorials/03_graph
   tutorials/04_custom_module

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/parameters   
   reference/logging
   reference/checkpointing
   reference/contexts
   reference/batching
   reference/randomness
   reference/containers
   reference/utils
   reference/delay_and_normalizer
   reference/ppo_adapter

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/networks
   api/algorithms
   api/wrappers
   api/jax_dataclass
