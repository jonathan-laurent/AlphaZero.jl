"""
    EnvOracles

This module provides implementations for environment oracles in the context of batched
Monte Carlo Tree Search (MCTS) algorithms. Environment oracles are used to guide the
exploration and exploitation behavior of the MCTS algorithm.

The module defines two main types of environment oracles:

1. `uniform_env_oracle`: Provides a uniform policy and value prior across all actions for a
    given environment.
2. `neural_network_env_oracle`: Uses a neural network to provide a learned policy and value
    prior for a given environment.

Both oracles return an `EnvOracle` object with the following structure:

- `init_fn`: A function that initializes the environment states, valid actions, and priors.
- `transition_fn`: A function that transitions the environments based on actions taken and
   updates the priors.

# Examples

```julia
# Using uniform_env_oracle
oracle = uniform_env_oracle()
initial_state = oracle.init_fn(envs)

# Using neural_network_env_oracle
nn = YourNN()  # replace with your own neural network that follows the FluxNetwork interface
oracle = neural_network_env_oracle(; nn=nn)
initial_state = oracle.init_fn(envs)
```
"""
module EnvOracles

using CUDA
using Flux

using ..BatchedEnvs
using ..BatchedMctsUtilities
using ..Network
using ..Util.Devices

export uniform_env_oracle, neural_network_env_oracle, get_valid_actions


"""
    validate_logits(logits, valid_actions)

Adds negative infinity to the logits of invalid actions so that when softmax is
applied, the probability of invalid actions is 0.
"""
function validate_logits(logits, valid_actions)
    action_mask = typemin(Float32) .* .!valid_actions
    return logits .+ action_mask
end

"""
    validate_value_prior!(value_prior, envs)

Sets the value prior to 0 for terminated environments. This is because by definition
the value function of a terminal state is 0.
"""
function validate_value_prior!(value_prior, envs)
    terminated_envs = BatchedEnvs.terminated.(envs)
    value_prior[1, :] = value_prior[1, :] .* .!terminated_envs
end

"""
    uniform_env_oracle()

Define an `EnvOracle` object with a uniform policy for any given environment.
See also [`EnvOracle`](@ref)
"""
function uniform_env_oracle()

    get_policy_prior(n_act, n_envs, device) = ones(Float32, device, (n_act, n_envs)) / n_act
    get_value_prior(n_envs, device) = zeros(Float32, device, n_envs)
    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function init_fn(envs)
        n_actions = BatchedEnvs.num_actions(eltype(envs))
        num_envs = length(envs)
        device = get_device(envs)

        valid_actions = get_valid_actions(envs)
        logits = get_policy_prior(n_actions, num_envs, device)
        logits = validate_logits(logits, valid_actions)
        policy_prior = Flux.softmax(logits; dims=1)

        return (;
            internal_states=envs,
            valid_actions=valid_actions,
            logit_prior=logits,
            policy_prior=policy_prior,
            value_prior=get_value_prior(num_envs, device),
        )
    end

    function transition_fn(envs, action_ids)
        n_actions = BatchedEnvs.num_actions(eltype(envs))
        num_envs = length(envs)
        device = get_device(envs)

        act_info = act.(envs, action_ids)
        internal_states = get_state.(act_info)

        valid_actions = get_valid_actions(internal_states)
        logits = get_policy_prior(n_actions, num_envs, device)
        logits = validate_logits(logits, valid_actions)
        policy_prior = Flux.softmax(logits; dims=1)

        return (;
            internal_states=internal_states,
            rewards=Float32.(get_reward.(act_info)),
            terminal=BatchedEnvs.terminated.(internal_states),
            valid_actions=valid_actions,
            player_switched=get_switched.(act_info),
            logit_prior=logits,
            policy_prior=policy_prior,
            value_prior=get_value_prior(num_envs, device),
        )
    end

    return EnvOracle(; init_fn, transition_fn)
end

"""
    neural_network_env_oracle(; nn::Net) where Net <: FluxNetwork

Define an `EnvOracle` object with a Neural-Network-guided policy for any given environment.
See also [`EnvOracle`](@ref)
"""
function neural_network_env_oracle(; nn::Net) where Net <: FluxNetwork

    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function init_fn(envs)
        num_envs = length(envs)
        device = get_device(envs)

        states = zeros(Float32, device, BatchedEnvs.state_size(eltype(envs))..., num_envs)
        Devices.foreach(1:num_envs, device) do env_id
            @inbounds states[:, env_id] .= BatchedEnvs.vectorize_state(envs[env_id])
            return nothing
        end
        value_prior, logits = forward(nn, states, false)

        valid_actions = get_valid_actions(envs)
        logits = validate_logits(logits, valid_actions)
        policy_prior = Flux.softmax(logits; dims=1)
        validate_value_prior!(value_prior, envs)

        return (;
            internal_states=envs,
            valid_actions=valid_actions,
            logit_prior=logits,
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    function transition_fn(envs, aids)
        num_envs = length(envs)
        device = get_device(envs)

        act_info = BatchedEnvs.act.(envs, aids)
        new_envs = get_state.(act_info)

        states = zeros(Float32, device, BatchedEnvs.state_size(eltype(envs))..., num_envs)
        Devices.foreach(1:num_envs, device) do env_id
            @inbounds states[:, env_id] .= BatchedEnvs.vectorize_state(new_envs[env_id])
            return nothing
        end
        value_prior, logits = forward(nn, states, false)

        valid_actions = get_valid_actions(new_envs)
        logits = validate_logits(logits, valid_actions)
        policy_prior = Flux.softmax(logits; dims=1)
        validate_value_prior!(value_prior, new_envs)

        return (;
            internal_states=new_envs,
            rewards=Float32.(get_reward.(act_info)),
            terminal=BatchedEnvs.terminated.(new_envs),
            valid_actions=valid_actions,
            player_switched=get_switched.(act_info),
            logit_prior=logits,
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    return EnvOracle(; init_fn, transition_fn)
end

end
