module BatchedMctsUtilities

using Base: @kwdef
using StaticArrays

import Base.Iterators.map as imap

using ..BatchedEnvs
using ..Util.Devices


export EnvOracle
export GumbelMctsConfig, AlphaZeroMctsConfig, is_gumbel_mcts_config
export get_valid_actions


# # Environment Oracle

"""
    EnvOracle{I <: Function, T <: Function}(; init_fn, transition_fn)

The `EnvOracle` struct serves as an abstraction layer that encapsulates environment
dynamics for Monte Carlo Tree Search (MCTS) procedures. It is specifically engineered to
facilitate batch operations conducive for parallel computations across multiple
computational units, such as GPUs and CPUs.

## Fields

- `init_fn::I`: Function for environment initialization.
- `transition_fn::T`: Function for state transition given actions.

### init_fn

The `init_fn` function receives an array of environment objects and returns a
NamedTuple with the following fields:

- `internal_states`: An array-like data structure that encapsulates the internal state
   representation, optimized for the MCTS algorithm and compatible with the `transition_fn`.
   For AlphaZero it should be the environments themselves, for MuZero it can be state
   representations.
- `valid_actions`: A Boolean array with dimensions `(num_actions, num_envs)`, denoting the
   admissibility of actions in each state.
- `logit_prior`: An `AbstractArray{Float32,2}` with dimensions `(num_actions, num_envs)`.
   Represents the unnormalized logarithm of action probabilities.
- `policy_prior`: A probability array with dimensions `(num_actions, num_envs)`, denoting
   the prior belief over actions.
- `value_prior`: An `AbstractVector{Float32}` containing prior values for states with
   dimensions `(num_envs,)`. In most environments, it's advised to constrain the values
   within `[-1, 1]`.

### transition_fn

The `transition_fn` function takes an array of internal states and an array of action
identifiers as inputs and returns a NamedTuple consisting of:

- `internal_states`: New environments (AlphaZero) or updated state representations (MuZero).
- `rewards`: An array of `Float32` indicating the immediate rewards resultant from the
   transitions. Dimensions: `(num_envs,)`.
- `terminal`: A Boolean array signifying whether the resultant states are terminal.
   Dimensions: `(num_envs,)`.
- `player_switched`: A Boolean array indicating if the player has changed during the
   transition. Dimensions: `(num_envs,)`.
- `valid_actions`, `logit_prior`, `policy_prior`, `value_prior`: Similar to `init_fn`,
   with the difference that these fields are now computed for the new states. Note that
   these values will be disregarded if the new states are terminal states.

### Constraints

- All arrays must be of type `isbits` to ensure GPU compatibility.
- Last dimension of all arrays should correspond to the batch dimension `(num_envs)`.

## Constraints on Computational Backends

- Depending on the device specified in the `MctsConfig`, `init_fn` and `transition_fn` can
  receive either a CPU-based array (`Array`) or a GPU-based array (`CuArray`).

## Examples of Internal State Encodings

- In AlphaZero the `internal_states` field should be the of the same type as the `envs`
  argument. That is, a batch of environments structures.
- In MuZero, the `internal_states` field should be of the type `AbstractArray{Float32, N}`,
  where the first (N-1) dimensions represent the latent state, and the last dimension
  corresponds to the batch dimension `(num_envs)`.
"""
@kwdef struct EnvOracle{I <: Function, T <: Function}
    init_fn::I
    transition_fn::T
end

# # MCTS Configurations definitions

"""Abstract type of a MCTS configuration."""
abstract type AbstractMctsConfig end

"""
    struct GumbelMctsConfig{Device, Oracle <: EnvOracle} <: AbstractMctsConfig
        device::Device
        oracle::Oracle
        num_simulations::Int
        num_considered_actions::Int = 8
        value_scale::Float32 = 0.1f0
        max_visit_init::Int = 50
    end

A batch, device-specific Gumbel MCTS MctsConfig that leverages an external `EnvOracle`.

# Keyword Arguments

- `device::Device`: Device on which the policy should preferably run (i.e. `CPU` or `GPU`).
- `oracle::Oracle`: Environment oracle handling the environment simulation and the state
   evaluation.
- `num_simulations::Int`: Number of simulations to run on the given MCTS `Tree`.
- `num_considered_actions::Int = 8`: Number of actions considered by Gumbel during
   exploration. Only the `num_considered_actions` actions with the highest scores will be
   explored. It should preferably be a power of 2.
- `value_scale::Float32 = 0.1f0`: `c_scale` parameter described in the Gumbel MCTS paper.
- `max_visit_init::Int = 50`: `c_visit` parameter described in the Gumbel MCTS paper.
"""
@kwdef struct GumbelMctsConfig{Device, Oracle <: EnvOracle} <: AbstractMctsConfig
    device::Device
    oracle::Oracle
    num_simulations::Int
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

"""
    struct AlphaZeroMctsConfig{Device, Oracle <: EnvOracle} <: AbstractMctsConfig
        device::Device
        oracle::Oracle
        num_simulations::Int
        c_puct::Float32 = 1.5f0
        alpha_dirichlet::Float32 = 0.15f0
        epsilon_dirichlet::Float32 = 0.25f0
        tau::Float32 = 1f0
        collapse_tau_move::Int = 30
    end

A batch, device-specific AlphaZero MCTS MctsConfig that leverages an external `EnvOracle`.

# Keyword Arguments

- `device::Device`: Device on which the policy should preferably run (i.e. `CPU` or `GPU`).
- `oracle::Oracle`: Environment oracle handling the environment simulation and the state
   evaluation.
- `num_simulations::Int`: Number of simulations to run on the given MCTS `Tree`.
- `c_puct::Float32 = 1.5f0`: `c_puct` parameter described in the AlphaZero MCTS paper.
- `alpha_dirichlet::Float32 = 0.15f0`: `alpha` parameter of the Dirichlet exploration noise.
- `epsilon_dirichlet::Float32 = 0.25f0`: `epsilon` weight of the Dirichlet noise.
- `tau::Float32 = 1f0`: Exploration temperature.
- `collapse_tau_move::Int = 30`: Number of actions after which the temperature tau is set
   to 0 in an episode.
"""
@kwdef struct AlphaZeroMctsConfig{Device, Oracle <: EnvOracle} <: AbstractMctsConfig
    device::Device
    oracle::Oracle
    num_simulations::Int
    c_puct::Float32 = 1.5f0
    alpha_dirichlet::Float32 = 0.15f0
    epsilon_dirichlet::Float32 = 0.25f0
    tau::Float32 = 1f0
    collapse_tau_move::Int = 30
end

""" Check if a given `AbstractMctsConfig` is a `GumbelMctsConfig`. """
is_gumbel_mcts_config(::GumbelMctsConfig) = true
is_gumbel_mcts_config(::AlphaZeroMctsConfig) = false

"""
    get_valid_actions(envs)

Returns an array of valid actions for each environment in `envs`.
The array has size `(n_actions, n_envs)`, and each entry is 1 if the
corresponding action is valid, and 0 otherwise.
"""
function get_valid_actions(envs)
    device = get_device(envs)
    num_envs = length(envs)
    n_actions = BatchedEnvs.num_actions(eltype(envs))
    valid_ids = DeviceArray(device)(Tuple.(CartesianIndices((n_actions, num_envs))))
    return map(valid_ids) do (action_id, batch_id)
        BatchedEnvs.valid_action(envs[batch_id], action_id)
    end
end

end
