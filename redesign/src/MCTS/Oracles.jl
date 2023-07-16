module EnvOracles

using CUDA

using ..BatchedEnvs
using ..BatchedMctsUtilities
using ..Network
using ..Util.Devices

export uniform_env_oracle, neural_network_env_oracle, get_valid_actions


function get_valid_actions(envs, device)
    n_actions = BatchedEnvs.num_actions(eltype(envs))
    valid_ids = DeviceArray(device)(Tuple.(CartesianIndices((n_actions, length(envs)))))
    return map(valid_ids) do (action_id, batch_id)
        BatchedEnvs.valid_action(envs[batch_id], action_id)
    end
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

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(envs, device),
            policy_prior=get_policy_prior(n_actions, num_envs, device),
            value_prior=get_value_prior(num_envs, device),
        )
    end

    function transition_fn(envs, action_ids)
        n_actions = BatchedEnvs.num_actions(eltype(envs))
        num_envs = length(envs)
        device = get_device(envs)

        act_info = act.(envs, action_ids)
        internal_states = get_state.(act_info)

        return (;
            internal_states=internal_states,
            rewards=Float32.(get_reward.(act_info)),
            terminal=BatchedEnvs.terminated.(internal_states),
            valid_actions=get_valid_actions(internal_states, device),
            player_switched=get_switched.(act_info),
            policy_prior=get_policy_prior(n_actions, num_envs, device),
            value_prior=get_value_prior(num_envs, device),
        )
    end

    return EnvOracle(; init_fn, transition_fn)
end


"""
    neural_network_env_oracle()

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

        states = zeros(Float32, device, BatchedEnvs.state_size(eltype(envs)), num_envs)
        Devices.foreach(1:num_envs, device) do env_id
            @inbounds states[:, env_id] .= BatchedEnvs.vectorize_state(envs[env_id])
            return nothing
        end
        value_prior, policy_prior = forward(nn, states)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(envs, device),
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    function transition_fn(envs, aids)
        num_envs = length(envs)
        device = get_device(envs)

        act_info = BatchedEnvs.act.(envs, aids)
        new_envs = get_state.(act_info)

        states = zeros(Float32, device, BatchedEnvs.state_size(eltype(envs)), num_envs)
        Devices.foreach(1:num_envs, device) do env_id
            @inbounds states[:, env_id] .= BatchedEnvs.vectorize_state(new_envs[env_id])
            return nothing
        end
        value_prior, policy_prior = forward(nn, states)

        return (;
            internal_states=new_envs,
            rewards=Float32.(get_reward.(act_info)),
            terminal=BatchedEnvs.terminated.(new_envs),
            valid_actions=get_valid_actions(envs, device),
            player_switched=get_switched.(act_info),
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    return EnvOracle(; init_fn, transition_fn)
end


end
