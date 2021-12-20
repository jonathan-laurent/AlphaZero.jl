using Graphs
using Plots, GraphRecipes
using AlphaZero
using GraphNeuralNetworks
import AlphaZero.GI


struct GameSpec <: GI.AbstractGameSpec
    verticies::Vector{Int}
    fadjlist::Matrix{T} where {T}
end

mutable struct GameEnv <: GI.AbstractGameEnv
    verticies::Vector{Int}
    fadjlist::Matrix{T} where {T}
    maskedActions::Vector{Bool}
    visitedVerticies::Vector{Int}
    finished::Bool
end

GI.spec(game::GameEnv) = GameSpec(game.verticies, game.fadjlist)

function GI.init(spec::GameSpec)
    return GameEnv(spec.verticies, spec.fadjlist, trues(length(spec.verticies)), Vector{Int}(), false)
end

function GI.set_state!(game::GameEnv, state)
    game.maskedActions = state.availableActions
    game.visitedVerticies = state.path
    any(game.maskedActions) || (game.finished = true)
    return
end

GI.two_players(::GameSpec) = false
GI.actions(a::GameSpec) = collect(range(1, length = length(a.verticies)))
GI.clone(g::GameEnv) = GameEnv(g.verticies, g.fadjlist, deepcopy(g.maskedActions), deepcopy(g.visitedVerticies), g.finished)
GI.current_state(g::GameEnv) = (path = g.visitedVerticies, availableActions = g.maskedActions)
GI.white_playing(::GameEnv) = true
GI.game_terminated(g::GameEnv) = g.finished
GI.available_actions(g::GameEnv) = g.verticies[g.maskedActions]
GI.actions_mask(game::GameEnv) = game.maskedActions

function GI.play!(g::GameEnv, vertex::Int)
    maskedActions = deepcopy(g.maskedActions)
    visitedVerticies = deepcopy(g.visitedVerticies)
    maskedActions[vertex] = false
    state = (path = push!(visitedVerticies, vertex), availableActions = maskedActions)
    GI.set_state!(g, state)
end

function GI.white_reward(g::GameEnv)
    isempty(g.visitedVerticies[1:end-1]) && (return 0.0)
    return -1 * sum(eachindex(g.visitedVerticies[1:end-1])) do vert
        g.fadjlist[g.visitedVerticies[vert+1], g.visitedVerticies[vert]]
    end
end

function GI.heuristic_value(g::GameEnv)
    return GI.white_reward(g)
end

function GI.render(g::GameEnv)
    nVerticies = length(g.verticies)
    graph = SimpleDiGraph(nVerticies, 0)
    foreach(enumerate(g.visitedVerticies[1:end-1])) do (idx, vert)
        add_edge!(graph, vert, g.visitedVerticies[idx+1])
    end
    graphplot(graph; curves = false)
end

function GI.graph_state(spec::GameSpec, state)
    if length(state.path) > 1
        modifiedAdacencyList = deepcopy(spec.fadjlist)
        foreach(enumerate(state.path[2:end])) do (idx, node)
            prevNode = state.path[idx]
            modifiedAdacencyList[prevNode, :] .= Inf
            modifiedAdacencyList[:, node] .= Inf
            modifiedAdacencyList[node, prevNode] = Inf
            modifiedAdacencyList[prevNode, node] = spec.fadjlist[prevNode, node]
        end
        return SimpleDiGraph(modifiedAdacencyList)
    else
        return SimpleDiGraph(spec.fadjlist)
    end
end