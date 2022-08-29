module Storage

import Base.length

using ..BatchedEnvs
using ..TrainableEnvOracles

export GameHistory, save, sample_position, length
export ReplayBuffer, save, get_sample

"""
    GameHistory

Buffers saving all training-related information about a game.
"""
struct GameHistory
    states::Vector{Vector{Float32}}
    actions::Vector{Int16}
    rewards::Vector{Float32}
    values::Vector{Float32}
    policies::Vector{Vector{Float32}}

    GameHistory() = new([], [], [], [], [])
end

function save(history::GameHistory, state, action, reward, value, policy)
    push!(history.states, state)
    push!(history.actions, action)
    push!(history.rewards, reward)
    push!(history.values, value)
    push!(history.policies, policy)

    return nothing
end

sample_position(history::GameHistory) = rand(eachindex(history.actions))
Base.length(history::GameHistory) = length(history.actions)

"""
    ReplayBuffer

Fixed-size buffer storing simulated games.
"""
struct ReplayBuffer
    games::Vector{GameHistory}
    size
    ReplayBuffer(size) = new([], size)
end

function save(replay_buffer::ReplayBuffer, game::GameHistory)
    if (length(replay_buffer.games) >= replay_buffer.size)
        popfirst!(replay_buffer.games)
    end
    push!(replay_buffer.games, game)
    return nothing
end

function save(replay_buffer::ReplayBuffer, games::AbstractArray{GameHistory})
    map(game -> save(replay_buffer, game), games)
    return nothing
end

function get_sample(
    replay_buffer::ReplayBuffer, trainable_oracle::TrainableEnvOracle, train_settings
)
    games = rand(replay_buffer.games, train_settings.batch_size)
    game_pos = map(sample_position, games)
    map(zip(games, game_pos)) do (game, pos)
        make_feature_and_target(game, trainable_oracle, pos, train_settings)
    end
end

end