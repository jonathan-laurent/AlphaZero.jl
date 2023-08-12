module ReplayBuffers

using EllipsisNotation
using Flux: onehotbatch

using ...Util.Devices

export Sample
export EpisodeBuffer, save!, empty_env_in_buffer!, compute_value_functions!
export ReplayBuffer, add!, to_array


"""
    Sample

A single sample of data from the environment.
"""
mutable struct Sample{NumStateDims}
    state::Array{Float32, NumStateDims}
    action::Int16
    reward::Float32
    switched::Bool
end

function Sample(state_size::Tuple)
    state = zeros(Float32, state_size...)
    action = 0
    reward = 0.0
    switched = false
    return Sample{length(state_size)}(state, action, reward, switched)
end


"""
    EpisodeBuffer

Buffer saving all training-related information episode-data of parallel environments.
Data will be saved in the form of Vectors on the CPU to avoid filling the GPU memory,
as it's needed to fit as many parralel environments as possible.
"""
struct EpisodeBuffer
    num_envs::Int
    samples::Vector{Vector{Sample}}
end

Base.getindex(ep_buff::EpisodeBuffer, env_id::Int) = ep_buff.samples[env_id]

function EpisodeBuffer(num_envs::Int)
    samples = [Vector{Sample}([]) for _ in 1:num_envs]
    return EpisodeBuffer(num_envs, samples)
end

function save!(ep_buff, states, actions, rewards, switches)
    states = copy_to_CPU(states)
    actions = copy_to_CPU(actions)
    rewards = copy_to_CPU(rewards)

    map(1:ep_buff.num_envs) do env_id
        sample = Sample(
            Array(states[env_id]),
            actions[env_id],
            rewards[env_id],
            switches[env_id]
        )
        push!(ep_buff.samples[env_id], sample)
    end
end

function compute_value_functions!(ep_buff, env_id, γ)
    end_index = length(ep_buff[env_id])
    reward_to_go = 0
    for idx in end_index:-1:1
        reward_to_go = γ * reward_to_go + ep_buff[env_id][idx].reward
        ep_buff[env_id][idx].reward = reward_to_go
        ep_buff[env_id][idx].switched && (reward_to_go = -reward_to_go)
    end
end

function empty_env_in_buffer!(ep_buff::EpisodeBuffer, env_id)
    ep_buff.samples[env_id] = Vector{Sample}([])
end


"""
    ReplayBuffer

Replay Buffer used to save all training-related data of terminated episodes.
"""
struct ReplayBuffer
    num_envs::Int
    max_steps::Int
    state_size::Tuple
    num_actions::Int
    states::Vector{Vector{Array{Float32}}}
    actions::Vector{Vector{Int16}}
    rewards::Vector{Vector{Float32}}
    most_recent_idx::Vector{Int}
end

function ReplayBuffer(
    num_envs::Int,
    max_steps_per_env::Int,
    state_size::Tuple,
    num_actions::Int
)
    states = [Vector{Array{Float32, length(state_size)}}([]) for _ in 1:num_envs]
    actions = [Vector{Int16}([]) for _ in 1:num_envs]
    rewards = [Vector{Float32}([]) for _ in 1:num_envs]
    most_recent_idx = zeros(Int, num_envs)
    return ReplayBuffer(num_envs, max_steps_per_env, state_size, num_actions,
                        states, actions, rewards, most_recent_idx)
end

function Base.length(rp_buff::ReplayBuffer)
    return sum(map(env_id -> length(rp_buff.states[env_id]), 1:rp_buff.num_envs))
end

function _append!(rp_buff, ep_buff, ep_range, env_id)
    append!(rp_buff.states[env_id], map(t -> ep_buff[env_id][t].state, ep_range))
    append!(rp_buff.actions[env_id], map(t -> ep_buff[env_id][t].action, ep_range))
    append!(rp_buff.rewards[env_id], map(t -> ep_buff[env_id][t].reward, ep_range))
end

function _overwrite!(rp_buff, rp_range, ep_buff, ep_range, env_id)
    rp_buff.states[env_id][rp_range] .= map(t -> ep_buff[env_id][t].state, ep_range)
    rp_buff.actions[env_id][rp_range] .= map(t -> ep_buff[env_id][t].action, ep_range)
    rp_buff.rewards[env_id][rp_range] .= map(t -> ep_buff[env_id][t].reward, ep_range)
end

function add!(rp_buff::ReplayBuffer, ep_buff::EpisodeBuffer, env_id::Int)
    current_buffer_len = length(rp_buff.states[env_id])
    episode_len = length(ep_buff[env_id])

    # buffer is full, overwrite oldest data
    if current_buffer_len == rp_buff.max_steps
        most_recent_idx = rp_buff.most_recent_idx[env_id]

        if most_recent_idx + episode_len > rp_buff.max_steps
            steps_until_max = rp_buff.max_steps - most_recent_idx
            rp_range = (most_recent_idx + 1):rp_buff.max_steps
            ep_range = 1:steps_until_max
            _overwrite!(rp_buff, rp_range, ep_buff, ep_range, env_id)

            steps_from_start = episode_len - steps_until_max
            rp_range = 1:steps_from_start
            ep_range = (steps_until_max + 1):episode_len
            _overwrite!(rp_buff, rp_range, ep_buff, ep_range, env_id)

            rp_buff.most_recent_idx[env_id] = steps_from_start
        else
            most_recent_idx = rp_buff.most_recent_idx[env_id]
            rp_range = (most_recent_idx + 1):(most_recent_idx + episode_len)
            ep_range = 1:episode_len
            _overwrite!(rp_buff, rp_range, ep_buff, ep_range, env_id)

            rp_buff.most_recent_idx[env_id] = most_recent_idx + episode_len
        end

    # buffer is not full yet, append data
    else
        if current_buffer_len + episode_len > rp_buff.max_steps
            steps_until_max = rp_buff.max_steps - current_buffer_len
            ep_range = 1:steps_until_max
            _append!(rp_buff, ep_buff, ep_range, env_id)

            steps_from_start = episode_len - steps_until_max
            rp_range = 1:steps_from_start
            ep_range = (steps_until_max + 1):episode_len
            _overwrite!(rp_buff, rp_range, ep_buff, ep_range, env_id)

            rp_buff.most_recent_idx[env_id] = steps_from_start
        else
            ep_range = 1:episode_len
            _append!(rp_buff, ep_buff, ep_range, env_id)

            rp_buff.most_recent_idx[env_id] = current_buffer_len + episode_len
        end
    end
end

function to_array(rp_buff::ReplayBuffer, device::Device)
    num_samples = length(rp_buff)
    cat_dim = length(rp_buff.state_size) + 1

    # put all the states in a single array of size (state_size..., batch)
    states = zeros(Float32, rp_buff.state_size..., num_samples)
    actions = zeros(Int16, num_samples)
    rewards = zeros(Float32, 1, num_samples)

    current_pos = 0
    for env_id in 1:rp_buff.num_envs
        env_num_samples = length(rp_buff.states[env_id])
        (env_num_samples == 0) && continue
        target_range = (current_pos + 1):(current_pos + env_num_samples)

        states[.., target_range] = cat(rp_buff.states[env_id]..., dims=cat_dim)
        actions[target_range] = cat(rp_buff.actions[env_id]..., dims=cat_dim)
        rewards[1, target_range] = cat(rp_buff.rewards[env_id]..., dims=cat_dim)

        current_pos += env_num_samples
    end

    # convert to arrays in target device
    states = DeviceArray(device)(states)
    actions = DeviceArray(device)(actions)
    rewards = DeviceArray(device)(rewards)

    # one-hot encode the actions
    actions = onehotbatch(actions, 1:rp_buff.num_actions)

    return states, actions, rewards
end

end
