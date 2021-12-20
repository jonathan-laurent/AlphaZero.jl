using Graphs
using Plots, GraphRecipes
using AlphaZero
using GraphNeuralNetworks
import AlphaZero.GI


struct GameSpec <: GI.AbstractGameSpec
    verticies::Vector{Int}
    fadjlist::Vector{Vector}
end

mutable struct GameEnv <: GI.AbstractGameEnv
    verticies::Vector{Int}
    fadjList::Vector{Vector}
    maskedActions::Vector{Bool}
    visitedVerticies::Vector{Int}
    finished::Bool
end

GI.spec() = GameSpec()

function GI.init(spec::GameSpec)
    return GameEnv(spec.verticies, spec.fadjlist, trues(length(spec.verticies)), Vector{Int}(), false)
end

function GI.set_state!(g::GameEnv, currentVertex)
    g.maskedActions[currentVertex] = false
    push!(g.visitedVerticies, currentVertex)
    any(g.maskedActions) || (g.finished = true)
    return
end

two_players(::GameSpec) = false
GI.actions(a::GameSpec) = collect(Base.OneTo(size(a.eccentricities)[1]))
GI.clone(g::GameEnv) = GameEnv(g.verticies, g.fadjList, deepcopy(g.maskedActions), deepcopy(g.visitedVerticies), g.finished)
GI.play!(g::GameEnv, vertex::Int) = GI.set_state!(g, vertex)
GI.current_state(g::GameEnv) = (path = g.visitedVerticies,)
GI.white_playing(::GameEnv) = true
GI.game_terminated(g::GameEnv) = g.finished
GI.available_actions(g::GameEnv) = g.verticies[g.maskedActions]

function GI.white_reward(g::GameEnv)
    return -1 * sum(eachindex(g.visitedVerticies[1:end-1])) do vert
        g.fadjList[g.visitedVerticies[vert+1], g.visitedVerticies[vert]]
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

function modelCreator(din, d, dout)
    return GNNChain(GCNConv(din => d),
        BatchNorm(d),
        x -> relu.(x),
        GCNConv(d => d, relu),
        Dropout(0.5),
        Dense(d, dout)
    )
end

# function vectorize_state(a::AbstractGameSpec, state)
#     return state
# end

function graph_state(spec::GameSpec, state)
    adjacencyList = spec.fadjlist[state.visitedVerticies, state.visitedVerticies]
    initialGraph = SimpleDiGraph(adjacencyList)

    verts = state.visitedVerticies
    nextVerts = circshift(state.visitedVerticies, -1)
    foreach(zip(verts, nextVerts)) do (v1, v2)
        add_edge!(initialGraph, v1, v2) || @warn "$v1 and $v2 cannot be connected"
    end
    return initialGraph
end