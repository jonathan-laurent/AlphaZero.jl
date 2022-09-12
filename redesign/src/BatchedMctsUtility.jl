module BatchedMctsUtility

using ..BatchedEnvs
using ..Util.Devices

include("./Tests/Common/BitwiseTicTacToe.jl")
using .BitwiseTicTacToe

using Base: @kwdef

export Policy, EnvOracle, check_oracle, UniformTicTacToeEnvOracle

# # Environment Oracle

"""
    EnvOracle(; init_fn, transition_fn)

An environment oracle is defined by two functions: `init_fn` and `transition_fn`. These
functions operate on batches of states directly, enabling efficient parallelization on GPU,
CPU or both.

# The `init_fn` function

`init_fn` takes a vector of environment objects as an argument. Environment objects are of
the same type as those passed to the `explore` and `gumbel_explore` functions. The
`init_fn` function returns a named-tuple of same-size arrays with the following fields:

- `internal_states`: internal representations of the environment states as used by MCTS and
   manipulated by `transition_fn`. `internal_states` must be a single or multi-dimensional
   array whose last dimension is a batch dimension (see examples below). `internal_states`
   must be `isbits` if it is desired to run `BatchedMcts` on `GPU`.
- `valid_actions`: a vector of booleans with dimensions `num_actions` and `batch_id`
   indicating which actions are valid to take (this is disregarded in MuZero).
- `policy_prior`: the policy prior for each state as an `AbstractArray{Float32,2}` with
   dimensions `num_actions` and `batch_id` and value within [0, 1]. An action with
   `policy_prior` of `0` corresponds to the worst possible one or an illegal action. On the
   other side, a value of `1` is associated with a winning action.
- `value_prior`: the value prior for each state as an `AbstractVector{Float32}` and value
   within [-1, 1]. In the same way than `policy_prior`, -1 and 1 respectively correspond to 
   a bad and a good position.

# The `transition_fn` function

`transition_fn` takes as arguments a vector of internal states (as returned by `init_fn`)
along with a vector of action ids. Action ids consist in integers between 1 and
`num_actions` and are valid indices for `policy_prior` and `value_prior`. 
    
Note that `init_fn` will always receive the same array as the one passed to `explore` or
`gumbel_explore` as `envs` (which should be a CPU `Array`). But it's a bit more tricky for
`transition_fn`. It may receive both CPU `Array` or GPU `CuArray` depending on the device
specified in `Policy`. To handle both more easily look at `Util.Devices` and how it 
is used in `UniformTicTacToeEnvOracle`.

In the context of a `Policy` on the GPU, `transition_fn` can both return a CPU `Array` or a
GPU `CuArray`. The `CuArray` is more adapted as it will prevent memory transfers, but both
works.

The `transition_fn` function returns a named-tuple of arrays:

- `internal_states`: new states reached after executing the proposed actions (see
   `init_fn`).
- `rewards`: vector of `Float32` indicating the intermediate rewards collected during the
   transitions.
- `terminal`: vector of booleans indicating whether or not the reached states are terminal
   (this is always `false` in MuZero).
- `player_switched`: vector of booleans indicating whether or not the current player
   switched during the transition (always `true` in many board games).
- `valid_actions`, `policy_prior`, `value_prior`: same as for `init_fn`.


# Examples of internal state encodings

- When using AlphaZero on board games such as tictactoe, connect-four, Chess or Go, internal
  states are exact state encodings (since AlphaZero can access an exact simulator). The
  `internal_states` field can be made to have type `AbstractArray{Float32, 4}` with
  dimensions `player_channel`, `board_width`, `board_height` and `batch_id`. Alternatively,
  one can use a one-dimensional vector with element type `State` where
  `Base.isbitstype(State)`. The latter representation may be easier to work with when
  broadcasting non-batched environment implementations on GPU (see
  `Tests.Common.BitwiseTicTacToe.BitwiseTicTacToe` for example).
- When using MuZero, the `internal_states` field typically has the type `AbstractArray{Float32,
  2}` where the first dimension corresponds to the size of latent states and the second
  dimension is the batch dimension.

See also [`check_oracle`](@ref), [`UniformTicTacToeEnvOracle`](@ref)
"""
@kwdef struct EnvOracle{I<:Function,T<:Function}
    init_fn::I
    transition_fn::T
end

"""
    check_keys(keys, ref_keys)

Check that the two lists of symbols, `keys` and `ref_keys`, are identical.

Small utilities used  in `check_oracle` to compare keys of named-tuple.
"""
function check_keys(keys, ref_keys)
    return Set(keys) == Set(ref_keys)
end

"""
    check_oracle(::EnvOracle, env)

Perform some sanity checks to see if an environment oracle is correctly specified on a
given environment instance.

A list of environments `envs` must be specified, along with the `EnvOracle` to check.

Return `nothing` if no problems are detected. Otherwise, helpful error messages are raised.
More precisely, `check_oracle` verifies the keys of the returned named-tuples from `init_fn`
a `transition_fn` and the types and dimensions of their lists.

See also [`EnvOracle`](@ref)
"""
function check_oracle(oracle::EnvOracle, envs)
    B = length(envs)

    init_res = oracle.init_fn(envs)
    # Named-tuple check
    @assert check_keys(
        keys(init_res), (:internal_states, :valid_actions, :policy_prior, :value_prior)
    ) "The `EnvOracle`'s `init_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, valid_actions, policy_prior, " *
        "value_prior."

    # Type and dimensions check
    size_valid_actions = size(init_res.valid_actions)
    A, _ = size_valid_actions
    @assert (size_valid_actions == (A, B) && eltype(init_res.valid_actions) == Bool) "The " *
        "`init_fn`'s function should return a `valid_actions` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Bool`."
    size_policy_prior = size(init_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(init_res.policy_prior) == Float32) "The " *
        "`init_fn`'s function should return a `policy_prior` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Float32`."
    @assert (length(init_res.value_prior) == B && eltype(init_res.value_prior) == Float32) "The " *
        "`init_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    aids = [
        findfirst(init_res.valid_actions[:, bid]) for
        bid in 1:B if any(init_res.valid_actions[:, bid])
    ]
    envs = [env for (bid, env) in enumerate(envs) if any(init_res.valid_actions[:, bid])]

    transition_res = oracle.transition_fn(envs, aids)
    # Named-tuple check
    @assert check_keys(
        keys(transition_res),
        (
            :internal_states,
            :rewards,
            :terminal,
            :valid_actions,
            :player_switched,
            :policy_prior,
            :value_prior,
        ),
    ) "The `EnvOracle`'s `transition_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, rewards, terminal, valid_actions, " *
        "player_switched, policy_prior, value_prior."

    # Type and dimensions check
    @assert (
        length(transition_res.rewards) == B && eltype(transition_res.rewards) == Float32
    ) "The `transition_fn`'s function should return a `rewards` vector of length " *
        "`batch_id` and of type `Float32`."
    @assert (
        length(transition_res.terminal) == B && eltype(transition_res.terminal) == Bool
    ) "The `transition_fn`'s function should return a `terminal` vector of length " *
        "`batch_id` and of type `Bool`."
    size_valid_actions = size(transition_res.valid_actions)
    @assert (size_valid_actions == (A, B) && eltype(transition_res.valid_actions) == Bool) "The `" *
        "transition_fn`'s function should return a `valid_actions` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Bool`."
    @assert (
        length(transition_res.player_switched) == B &&
        eltype(transition_res.player_switched) == Bool
    ) "The `transition_fn`'s function should return a `player_switched` vector of length " *
        "`batch_id`, and of type `Bool`."
    size_policy_prior = size(transition_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(transition_res.policy_prior) == Float32) "The " *
        "`transition_fn`'s function should return a `policy_prior` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Float32`."
    @assert (
        length(transition_res.value_prior) == B &&
        eltype(transition_res.value_prior) == Float32
    ) "The `transition_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    return nothing
end

# ## Example Environment Oracle
# ### Tic-Tac-Toe Environment Oracle with a Uniform policy

"""
    UniformTicTacToeEnvOracle()

Define an `EnvOracle` object with a uniform policy for the game of Tic-Tac-Toe.

This oracle environment is a wrapper around the `Tests.Common.BitwiseTicTacToeEnv`.
It can be both used on `CPU` & `GPU`.

It was inspired by the RL.jl library. For more details, check out their documentation:
https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TicTacToeEnv

See also [`EnvOracle`](@ref)
"""
function UniformTicTacToeEnvOracle()
    A = 9 # Number of case in the Tic-Tac-Toe grid

    get_policy_prior(A, B, device) = ones(Float32, device, (A, B)) / A
    get_value_prior(B, device) = zeros(Float32, device, B)

    function get_valid_actions(A, B, envs)
        device = get_device(envs)

        valid_ids = DeviceArray(device)(Tuple.(CartesianIndices((A, B))))
        return map(valid_ids) do (aid, bid)
            valid_action(envs[bid], aid)
        end
    end

    function init_fn(envs)
        B = length(envs)
        device = get_device(envs)

        @assert B > 0
        @assert all(@. num_actions(envs) == A)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(A, B, envs),
            policy_prior=get_policy_prior(A, B, device),
            value_prior=get_value_prior(B, device),
        )
    end

    # Utilities to access elements returned by `act` in `transition_fn`.
    # `act` return a tuple with the following form:
    #   (state, (; reward, switched))
    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function transition_fn(envs, aids)
        B = size(envs)[end]
        device = get_device(envs)

        @assert all(valid_action.(envs, aids)) "Tried to play an illegal move"

        act_info = act.(envs, aids)
        player_switched = get_switched.(act_info)
        rewards = Float32.(get_reward.(act_info))
        internal_states = get_state.(act_info)

        return (;
            internal_states,
            rewards,
            terminal=terminated.(internal_states),
            valid_actions=get_valid_actions(A, B, internal_states),
            player_switched,
            policy_prior=get_policy_prior(A, B, device),
            value_prior=get_value_prior(B, device),
        )
    end
    return EnvOracle(; init_fn, transition_fn)
end

# # Policy definition

"""
    Policy{Device, Oracle<:EnvOracle}(; 
        device::Device
        oracle::Oracle
        num_simulations::Int = 64
        num_considered_actions::Int = 8
        value_scale::Float32 = 0.1f0
        max_visit_init::Int = 50
    )

A batch, device-specific MCTS Policy that leverages an external `EnvOracle`.


# Keyword Arguments

- `device::Device`: device on which the policy should preferably run (i.e. `CPU` or `GPU`).
- `oracle::Oracle`: environment oracle handling the environment simulation and the state
   evaluation.
- `num_simulations::Int = 64`: number of simulations to run on the given Mcts `Tree`.
- `num_considered_actions::Int = 8`: number of actions considered by Gumbel during
   exploration. Only the `num_considered_actions` actions with the highest scores will be
   explored. It should preferably be a power of 2.
- `value_scale::Float32 = 0.1f0`: multiplying coefficient to weight the qvalues against the
   prior probabilities during exploration. Prior probabilities have, by default, a
   decreasing weight when the number of visits increases.
- `max_visit_init::Int = 50`: artificial increase of the number of visits to weight qvalue
   against prior probabilities on low visit count.

# Notes

The attributes `num_considered_actions`, `value_scale` and `max_visit_init` are specific to
the Gumbel implementation.
"""
@kwdef struct Policy{Device,Oracle<:EnvOracle}
    device::Device
    oracle::Oracle
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

end
