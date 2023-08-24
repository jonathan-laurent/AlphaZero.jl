module ReplayBuffers

using EllipsisNotation
using Flux

export EpisodeBuffer, save!, empty_env_in_buffer!, compute_value_functions!
export ReplayBuffer, add!, to_array


"""
    EpisodeBuffer

Buffer saving all training-related information episode-data of parallel environments.
Data will be saved in the form of Vectors on the CPU to avoid filling the GPU memory,
as it's needed to fit as many parralel environments as possible.
"""
mutable struct EpisodeBuffer
    states::Array{Float32}
    actions::Array{Int16}
    rewards::Array{Float32}
    switches::Array{Bool}
    ep_lengths::Array{Int}
end

state_dim(ep_buff::EpisodeBuffer) = size(ep_buff.states)[1:(end - 2)]
horizon(ep_buff::EpisodeBuffer) = size(ep_buff.states)[end - 1]
num_envs(ep_buff::EpisodeBuffer) = size(ep_buff.states)[end]

function EpisodeBuffer(num_envs::Int, state_size::Tuple, initial_horizon::Int = 10)
    states = zeros(Float32, state_size..., initial_horizon, num_envs)
    actions = zeros(Int16, initial_horizon, num_envs)
    rewards = zeros(Float32, initial_horizon, num_envs)
    switches = zeros(Bool, initial_horizon, num_envs)
    ep_lengths = zeros(Int, num_envs)
    return EpisodeBuffer(states, actions, rewards, switches, ep_lengths)
end

function _increase_horizon(ep_buff)
    current_horizon = horizon(ep_buff)
    new_horizon = 2current_horizon

    new_states = zeros(Float32, state_dim(ep_buff)..., new_horizon, num_envs(ep_buff))
    new_states[.., 1:current_horizon, :] .= ep_buff.states[.., 1:current_horizon, :]

    new_actions = zeros(Int16, new_horizon, num_envs(ep_buff))
    new_actions[1:current_horizon, :] .= ep_buff.actions[1:current_horizon, :]

    new_rewards = zeros(Float32, new_horizon, num_envs(ep_buff))
    new_rewards[1:current_horizon, :] .= ep_buff.rewards[1:current_horizon, :]

    new_switches = zeros(Bool, new_horizon, num_envs(ep_buff))
    new_switches[1:current_horizon, :] .= ep_buff.switches[1:current_horizon, :]

    ep_buff.states = new_states
    ep_buff.actions = new_actions
    ep_buff.rewards = new_rewards
    ep_buff.switches = new_switches
end

function save!(ep_buff, states::AbstractVector, actions, rewards, switches)
    # if the buffer is full, increase the horizon
    (maximum(ep_buff.ep_lengths) == horizon(ep_buff)) && _increase_horizon(ep_buff)

    # transfer data to CPU
    states, actions, rewards, switches = map(Array, (states, actions, rewards, switches))

    # save data
    ep_buff.ep_lengths .+= 1
    map(1:num_envs(ep_buff)) do env_id
        save_index = ep_buff.ep_lengths[env_id]
        ep_buff.states[.., save_index, env_id] .= states[env_id]
        ep_buff.actions[save_index, env_id] = actions[env_id]
        ep_buff.rewards[save_index, env_id] = rewards[env_id]
        ep_buff.switches[save_index, env_id] = switches[env_id]
    end
end

function compute_value_functions!(ep_buff, env_id, γ)
    end_index = ep_buff.ep_lengths[env_id]
    reward_to_go = 0
    for idx in end_index:-1:1
        reward_to_go = γ * reward_to_go + ep_buff.rewards[idx, env_id]
        ep_buff.rewards[idx, env_id] = reward_to_go
        ep_buff.switches[idx, env_id] && (reward_to_go = -reward_to_go)
    end
end

function empty_env_in_buffer!(ep_buff::EpisodeBuffer, env_id)
    ep_buff.ep_lengths[env_id] = 0
end


"""
    ReplayBuffer

Replay Buffer used to save all training-related data of terminated episodes.
"""
mutable struct ReplayBuffer
    max_size::Int
    num_actions::Int
    states::Array{Float32}
    actions::Array{Int16}
    values::Array{Float32}
    current_size::Int
    most_recent_pos::Int
end

function ReplayBuffer(max_size::Int, state_size::Tuple, num_actions::Int)
    states = zeros(Float32, state_size..., max_size)
    actions = zeros(Int16, max_size)
    values = zeros(Float32, 1, max_size)
    return ReplayBuffer(max_size, num_actions, states, actions, values, 0, 0)
end

Base.length(rp_buff::ReplayBuffer) = rp_buff.current_size

function _overwrite!(rp_buff, rp_buff_range, ep_buff, ep_buff_range, env_id)
    rp_buff.states[.., rp_buff_range] .= ep_buff.states[.., ep_buff_range, env_id]
    rp_buff.actions[rp_buff_range] .= ep_buff.actions[ep_buff_range, env_id]
    rp_buff.values[1, rp_buff_range] .= ep_buff.rewards[ep_buff_range, env_id]
end

function add!(rp_buff::ReplayBuffer, ep_buff::EpisodeBuffer, env_id::Int)
    episode_len = ep_buff.ep_lengths[env_id]
    save_pos = rp_buff.most_recent_pos

    # if current episodes exceeds buffer size, overwrite oldest data
    if save_pos + episode_len > rp_buff.max_size
        steps_until_max = rp_buff.max_size - save_pos
        rp_buff_range = (save_pos + 1):rp_buff.max_size
        ep_buff_range = 1:steps_until_max
        _overwrite!(rp_buff, rp_buff_range, ep_buff, ep_buff_range, env_id)

        steps_from_start = episode_len - steps_until_max
        rp_buff_range = 1:steps_from_start
        ep_buff_range = (steps_until_max + 1):episode_len
        _overwrite!(rp_buff, rp_buff_range, ep_buff, ep_buff_range, env_id)

        rp_buff.most_recent_pos = steps_from_start
        rp_buff.current_size = rp_buff.max_size

    # else, current episode can be placed without looping around the buffer
    else
        rp_buff_range = (save_pos + 1):(save_pos + episode_len)
        ep_buff_range = 1:episode_len
        _overwrite!(rp_buff, rp_buff_range, ep_buff, ep_buff_range, env_id)

        rp_buff.most_recent_pos += episode_len
        rp_buff.current_size = min(rp_buff.current_size + episode_len, rp_buff.max_size)
    end
end

function to_array(rp_buff::ReplayBuffer)
    # get data up to current size
    states = rp_buff.states[.., 1:length(rp_buff)]
    actions = rp_buff.actions[1:length(rp_buff)]
    state_values = rp_buff.values[:, 1:length(rp_buff)]

    # one-hot encode the actions
    actions = Flux.onehotbatch(actions, 1:rp_buff.num_actions)

    return states, actions, state_values
end

end
