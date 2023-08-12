module EnvOracles

using CUDA
using Flux

using ..BatchedEnvs
using ..BatchedMctsUtilities
using ..Network
using ..Util.Devices

export uniform_env_oracle, neural_network_env_oracle, get_valid_actions


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
    validate_value_prior(value_prior, envs)

Sets the value prior to 0 for terminated environments. This is because by definition
the value function of a terminal state is 0.
"""
function validate_value_prior(value_prior, envs)
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
        validate_value_prior(value_prior, envs)

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
        validate_value_prior(value_prior, new_envs)

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
