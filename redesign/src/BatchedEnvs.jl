"""
    BatchedEnvs

Interface for batchable environments that can be used in the Monte Carlo Tree Search (MCTS)
pipeline. This interface allows for easy integration with environments that are capable of
running on GPUs, facilitating efficient and parallelized computations.

## Notes

The environment implementations must be GPU-friendly, which means that:

1. The environment functions do not allocate memory dynamically.
2. The input/return types can be fully statically inferred (no dynamic dispatch).
3. The environment structure can be represented by an immutable `ibits` type.

## Functions

- `state_size(::Type{T})`: Returns the size of the state representation for a specific
   environment type.
- `num_actions(::Type{T})`: Returns the number of available actions in the environment for
   a specific type.
- `valid_action(env, action)`: Checks if an action is valid given the current state of an
   environment instance.
- `act(env, action)`: Applies an action to the environment and returns a new environment
   state along with any metadata.
- `terminated(env)`: Checks if the environment has reached a terminal state.
- `reset(env)`: Resets the environment to an initial state.
- `vectorize_state(env)`: Converts the environment state to a vector representation
   suitable for machine learning models.

"""
module BatchedEnvs


export state_size, num_actions, valid_action, act, terminated, vectorize_state


"""
    state_size(::Type{T}) where T

Returns the size of the state representation for environment type `T`, as a Tuple,
"""
function state_size end

"""
    num_actions(::Type{T}) where T

Returns the number of available actions for environment type `T`.
"""
function num_actions end

"""
    valid_action(env, action)

Checks if an `action` is valid given the current state of the environment instance `env`.
Returns `true` if the action is valid, `false` otherwise.
"""
function valid_action end

"""
    act(env, action)

Applies an `action` to the `env` and returns a new environment state along with any
metadata. The return type is a NamedTuple with the fields: (; reward, switched).
"""
function act end

"""
    terminated(env)

Returns `true` if the environment is in a terminal state, `false` otherwise.
"""
function terminated end

"""
    reset(env)

Returns the reseted environment `env` (state is an initial state).
"""
function reset end

"""
    vectorize_state(env)

Converts the current state of the environment instance `env` to a vectorized form.
The return type must be a StaticArray with the same size as specified in `state_size()`.
"""
function vectorize_state end

end
