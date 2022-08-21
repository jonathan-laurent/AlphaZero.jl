module GameHistoryModule

import Base.length

using ..BatchedEnvs

export GameHistory, save, sample_position, length

"""
    GameHistory

Buffers saving all training-related information about a game.
"""
struct GameHistory{GameEnv}
    images::Vector{GameEnv}
    actions::Vector{Int16}
    rewards::Vector{Float32}
    values::Vector{Float32}
    policies::Vector{Vector{Float32}}

    GameHistory(GameEnv) = new{GameEnv}([], [], [], [], [])
end

function save(history::GameHistory, image, action, reward, value, policy)
    push!(history.images, image)
    push!(history.actions, action)
    push!(history.rewards, reward)
    push!(history.values, value)
    push!(history.policies, policy)

    return nothing
end

sample_position(history::GameHistory) = rand(eachindex(history.actions))
Base.length(history::GameHistory) = length(history.actions)

end