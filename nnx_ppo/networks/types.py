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

@flax.struct.dataclass
class StatefulModuleOutput:
  next_state: ModuleState
  output: Any #TODO: narrow this down
  regularization_loss: jax.Array
  metrics: Dict[str, jax.Array]

class StatefulModule(abc.ABC, nnx.Module):
    '''Abstract base class for network modules and layers, specifying the interface 
    with the RL algorithm. Each module treated as stateful, with non-stateful modules
    having empty states.
    
    There are two types of module state. First, there is the state of the `nnx.Module`
    which follows the common NNX patterns. This state stores trainable parameters
    (of type `nnx.Param`) as well as any `RNGStreams` and other variables. This state
    is _not_ reset when the RL environment is reset.

    Second, there is an explicit carry state that is intended for stateful network
    layers, e.g. the hidden activations of RNNs. This state _is_ reset when the RL
    environment is reset.
    '''
    
    @abc.abstractmethod
    def __call__(self, module_state: ModuleState, obs) -> StatefulModuleOutput:
        '''Args:
          module_state (ModuleState):
            The current state of the module.
          obs (PyTree):
            Observation(s) to process. Leading dimension is batch.
        Returns (StatefulModuleOutput):
          next_state (ModuleState):
            The updated module state after processing `obs`. Should have the same
            structure as `module_state`.
          output (PyTree):
            The output of the network module. Should have batch as leading dimension.
          regularization_loss (float):
            A floating-point value that is added to the total loss. Useful for
            adding regularization or co-objectives to the networks.
          metrics (Dict[str, jax.Array]):
            Any logging metrics.
        '''

    def initialize_state(self, batch_size: int) -> ModuleState:
        '''Create a new state for this module.

        Args:
        batch_size (int): The batch size

        Returns:
        A `ModuleState` with `batch_size` as the leading dimension on any
        arrays in the tree. The default is an empty tuple, representing a
        stateless module.

        Notes:
        As JAX does not allow parallellized conditional evaluation, this 
        method will be called on every step. For environments where `done=True`,
        the corresponding network state will be replaced with the return value of
        this method.
        '''
        return ()
