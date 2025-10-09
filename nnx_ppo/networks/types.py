from typing import Union, Dict, Any, Tuple
import abc
import jax
import flax.struct
from flax import nnx

@flax.struct.dataclass
class PPONetworkOutput:
  actions: jax.Array
  loglikelihoods: jax.Array
  regularization_loss: jax.Array
  value_estimates: jax.Array
  metrics: Dict[str, Any]

ModuleState = Any #TODO: make into a proper type alias
class PPONetwork(abc.ABC):
  
  @abc.abstractmethod
  def __call__(self, network_state: ModuleState, obs) -> Tuple[Any, PPONetworkOutput]:
    '''Apply both actor and critic networks to the environment observation `obs`.
    
    Calling the critic on rollouts might be somewhat inefficient, but by grouping
    them into the same method, the interface is well-defined even if the actor and
    critic share stateful layers.

    Args:
      network_state (ModuleState): The activation state of the network plus any RNG keys
                                   used for stochastic layers.
      obs (PyTree): Observation of an environment state. It is a PyTree with the same
                    structure as env.step(...).obs
      done (jax.Array[bool]): A boolean array of shape (batch_size,) indicating whether
                              the episode ended at this observation. This should be used to
                              reset any stateful layers in the network.

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
  def initialize_state(self, batch_size: int) -> ModuleState:
    '''Create an initial state for the network, including RNG keys for any
    stochastic layers.

    Args:
      batch_size (int): The batch size of the returned state.
      '''
    
class StatefulModule(abc.ABC, nnx.Module):
    @abc.abstractmethod
    def __call__(self, module_state: ModuleState, obs) -> Tuple[Any, Any, Any]:
        pass

    def initialize_state(self, batch_size: int) -> ModuleState:
        return ()
