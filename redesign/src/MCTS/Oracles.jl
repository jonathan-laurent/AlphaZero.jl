module EnvOracles

using CUDA

using ..BatchedEnvs
using ..BatchedMctsUtilities
using ..Network
using ..Util.Devices

export uniform_env_oracle, neural_network_env_oracle


function get_valid_actions(n_actions, batch_size, envs, device)
    valid_ids = DeviceArray(device)(Tuple.(CartesianIndices((n_actions, batch_size))))
    return map(valid_ids) do (action_id, batch_id)
        valid_action(envs[batch_id], action_id)
    end
end


"""
    uniform_env_oracle()

Define an `EnvOracle` object with a uniform policy for any given environment.
See also [`EnvOracle`](@ref)
"""
function uniform_env_oracle()

    get_policy_prior(n_act, b_sz, device) = ones(Float32, device, (n_act, b_sz)) / n_act
    get_value_prior(b_sz, device) = zeros(Float32, device, b_sz)
    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function init_fn(envs)
        n_actions = CUDA.@allowscalar num_actions(envs[1])
        batch_size = length(envs)
        device = get_device(envs)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(n_actions, batch_size, envs, device),
            policy_prior=get_policy_prior(n_actions, batch_size, device),
            value_prior=get_value_prior(batch_size, device),
        )
    end

    function transition_fn(envs, action_ids)
        n_actions = CUDA.@allowscalar num_actions(envs[1])
        batch_size = length(envs)
        device = get_device(envs)

        act_info = act.(envs, action_ids)
        internal_states = get_state.(act_info)

        return (;
            internal_states=internal_states,
            rewards=Float32.(get_reward.(act_info)),
            terminal=terminated.(internal_states),
            valid_actions=get_valid_actions(n_actions, batch_size, internal_states, device),
            player_switched=get_switched.(act_info),
            policy_prior=get_policy_prior(n_actions, batch_size, device),
            value_prior=get_value_prior(batch_size, device),
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
        input_size = CUDA.@allowscalar state_size(envs[1])
        n_actions = CUDA.@allowscalar num_actions(envs[1])
        batch_size = length(envs)
        device = get_device(envs)

        states = DeviceArray(device){Float32}(undef, input_size, batch_size)
        Devices.foreach(1:batch_size, device) do batch_id
            @inbounds states[:, batch_id] .= vectorize_state(envs[batch_id])
            return nothing
        end
        value_prior, policy_prior = forward(nn, states)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(n_actions, batch_size, envs, device),
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    function transition_fn(envs, aids)
        input_size = CUDA.@allowscalar state_size(envs[1])
        n_actions = CUDA.@allowscalar num_actions(envs[1])
        batch_size = length(envs)
        device = get_device(envs)

        act_info = act.(envs, aids)
        internal_states = get_state.(act_info)

        states = DeviceArray(device){Float32}(undef, input_size, batch_size)
        Devices.foreach(1:batch_size, device) do batch_id
            @inbounds states[:, batch_id] .= vectorize_state(internal_states[batch_id])
            return nothing
        end
        value_prior, policy_prior = forward(nn, states)

        return (;
            internal_states=internal_states,
            rewards=Float32.(get_reward.(act_info)),
            terminal=terminated.(internal_states),
            valid_actions=get_valid_actions(n_actions, batch_size, envs, device),
            player_switched=get_switched.(act_info),
            policy_prior=policy_prior,
            value_prior=value_prior
        )
    end

    return EnvOracle(; init_fn, transition_fn)
end


end
