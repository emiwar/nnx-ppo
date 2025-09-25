from typing import Union, Dict, Any, Tuple
import abc
import jax
import flax.struct
from flax import nnx

# Note: raw_actions is the action before applying transforms (e.g., tanh). This is 
# needed because the inverse (e.g., arctan(action)) might not be numerically stable
# for large (absolute) values
@flax.struct.dataclass
class PPONetworkOutput:
  actions: jax.Array#Union[Dict, jax.Array]
  #raw_actions: Union[Dict, jax.Array]
  loglikelihoods: jax.Array#Union[float, Dict, jax.Array]
  regularization_loss: jax.Array#Union[float, Dict, jax.Array]
  value_estimates: jax.Array#Union[float, Dict, jax.Array]
  metrics: Dict[str, Any]

class AbstractPPOActorCritic(abc.ABC):
  
  @abc.abstractmethod
  def __call__(self, network_state, obs) -> Tuple[Any, PPONetworkOutput]:
    '''Apply both actor and critic networks to the environment observation `obs`.
    
    Calling the critic on rollouts might be somewhat inefficient, but by grouping
    them into the same method, the interface is well-defined even if the actor and
    critic share stateful layers.

    Args:
      network_state (PyTree): The activation state of the network plus any RNG keys
                              used for stochastic layers.
      obs (PyTree): Observation of an environment state. It is a PyTree with the same
                    structure as env.step(...).obs

    Returns: (new_state, network_output)
      new_state (PyTree): An updated network state, including split RNG keys.
      network_output (PPONetworkOutput):
        actions (PyTree): The actions to be sent back to the env
        loglikelihood (float): The log-likelihood for taking the action
        regularization_loss (float): Sum of all other regularizer terms. This will be added
                             to the RL loss target during training.
        value_estimates (float): The estimated value of `obs` by the critic network.
        metrics (dict): A dictionary with any intermediate values to log
    
    Note: for multi-agent / modular networks, the returned action, log likelihood, and
          and value estimate may be arbitrary PyTrees instead of floats.
    '''

  @abc.abstractmethod
  def initialize_state(self, rng: jax.Array) -> Dict:
    '''Initialize a new network state. The state should include all variables
    that get updated in `__call__`, for example hidden state of recurrent networks and
    RNG keys for stochastics layers. Weights and other trainable parameters should
    typically _not_ be included in the state. For state-less networks the state may
    be empty.'''

  @abc.abstractmethod
  def reset_state(self, network_state) -> Dict:
    '''Reset and return the network state. This method is called when the environment
    is reset due to termination or truncation. Hidden network states should likely be
    reinitialized, but rngs may be left unchanged.'''

class StatefulModule(abc.ABC, nnx.Module):
    @abc.abstractmethod
    def __call__(self, network_state, obs) -> Tuple[Any, Any, Any]:
        pass

    def initialize_state(self, rng: jax.Array):
        return ()

    def reset_state(self, network_state):
        return ()