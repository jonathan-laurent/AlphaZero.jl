using Graphs, SimpleWeightedGraphs
using Plots, GraphRecipes
using AlphaZero
using GraphNeuralNetworks
import AlphaZero.GI


struct GameSpec <: GI.AbstractGameSpec
    graph::Matrix{T} where T <: Number
end

function randGraph(graphSize::Int)
    graph = rand(graphSize, graphSize)
    foreach(enumerate(eachcol(graph))) do (idx, col)
        graph[idx, :] .= col
        graph[idx, idx] = 0
    end
    return graph
end

GameSpec() = GameSpec(randGraph(rand(collect(1:20))))

mutable struct GameEnv <: GI.AbstractGameEnv
    adjacencyList::Matrix{T} where T <: Number
    maskedActions::Vector{Bool} # Masked actions and visitedVerticies can be derived from graph, but are included for clarity
    visitedVerticies::Vector{Int}
    finished::Bool
end

GI.spec(game::GameEnv) = GameSpec(game.graph)

function GI.init(spec::GameSpec)
    return GameEnv(spec.graph, trues(length(spec.verticies)), Vector{Int}(), false)
end

function GI.set_state!(game::GameEnv, state)
    game.maskedActions = state.availableActions
    game.visitedVerticies = getPath(state.ndata.data)
    any(game.maskedActions) || (game.finished = true)
    return
end

function getPath(adjacencyMatrix::Matrix)
    startingVertex = 1
    madeConnection(vertex) = isone(count(!iszero, adjacencyMatrix[vertex,:]))
    isConnected(vertex) = isone(count(!iszero, adjacencyMatrix[:, vertex]))
    foreach(Base.OneTo(size(adjacencyMatrix)[1])) do idx
        if madeConnection(idx) && !isConnected(idx)
            startingVertex = idx
        end
    end
    path = [startingVertex]
    foreach(path) do vertex
        if madeConnection(vertex) || isConnected(vertex)
            push!(path, findfirst(!iszero, adjacencyMatrix[vertex,:]))
        end
    end
    return path
end

GI.two_players(::GameSpec) = false
GI.actions(a::GameSpec) = collect(range(1, length = length(a.verticies)))
GI.clone(g::GameEnv) = GameEnv(g.graph, deepcopy(g.maskedActions), deepcopy(g.visitedVerticies), g.finished)
GI.white_playing(::GameEnv) = true
GI.game_terminated(g::GameEnv) = g.finished
GI.available_actions(g::GameEnv) = g.verticies[g.maskedActions]
GI.actions_mask(game::GameEnv) = game.maskedActions

function GI.current_state(g::GameEnv)
    modifiedAdacencyList = deepcopy(g.graph.weights)
    if length(g.visitedVerticies) > 1
        foreach(enumerate(g.visitedVerticies[2:end])) do (idx, node)
            prevNode = state.path[idx]
            # Values are set to zero because SimpleWeightedGraphs discards edges with weights of zero.
            modifiedAdacencyList[prevNode, :] .= 0
            modifiedAdacencyList[:, node] .= 0
            modifiedAdacencyList[node, prevNode] = 0
            modifiedAdacencyList[prevNode, node] = spec.fadjlist[prevNode, node]
        end
    end
    return (ndata = (data = modifiedAdacencyList, x = ones(2)), availableActions = g.maskedActions)
end

function GI.play!(g::GameEnv, vertex::Int)
    maskedActions = deepcopy(g.maskedActions)
    visitedVerticies = deepcopy(g.visitedVerticies)
    maskedActions[vertex] = false
    index = (vertex, last(visitedVerticies))

    state = (path = push!(visitedVerticies, vertex), availableActions = maskedActions)
    GI.set_state!(g, state)
end

function GI.white_reward(g::GameEnv)
    isempty(g.visitedVerticies[1:end-1]) && (return 0.0)
    return -1 * sum(eachindex(g.visitedVerticies[1:end-1])) do vert
        g.graph.weights[g.visitedVerticies[vert+1], g.visitedVerticies[vert]]
    end
end
re
function GI.heuristic_value(g::GameEnv)
    return GI.white_reward(g)
end

GI.render(g::GameEnv) =  graphplot(g.graph; curves = false)

function GI.graph_state(spec::GameSpec, state)
    return state.ndata.data
end