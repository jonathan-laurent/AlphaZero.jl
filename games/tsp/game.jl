using Graphs, SimpleWeightedGraphs
using Plots, GraphRecipes
using AlphaZero
using GraphNeuralNetworks
import AlphaZero.GI


struct GameSpec <: GI.AbstractGameSpec
    gnnGraph
end

function randGraph(numVerticies::Int)
    graph = rand(Float32, numVerticies, numVerticies)
    foreach(enumerate(eachcol(graph))) do (idx, col)
        graph[idx, :] .= col
        graph[idx, idx] = 0
    end
    nodes = collect(1:numVerticies)
    sources = Vector(vec(repeat(nodes, 1, numVerticies - 1)'))
    targets = vcat(map(idx -> filter(val -> val != idx, nodes), nodes)...)
    weights = map(zip(sources, targets)) do (source, target)
        graph[source, target]
    end
    return GNNGraph(sources, targets, weights; ndata = (; x = ones(Float32, 1, numVerticies)))
end
GI.isgraph(::GameSpec) = true

# GameSpec() = GameSpec(randGraph(rand(collect(1:20))))
GameSpec() = GameSpec(randGraph(5))

mutable struct GameEnv <: GI.AbstractGameEnv
    gnnGraph
    maskedActions::Vector{Bool} # Masked actions and visitedVerticies can be derived from graph, but are included for clarity
    visitedVerticies::Vector{Int}
    finished::Bool
end
GI.spec(game::GameEnv) = GameSpec(game.gnnGraph)

function GI.init(spec::GameSpec)
    return GameEnv(spec.gnnGraph, trues(spec.gnnGraph.num_nodes), Vector{Int}([rand(Base.OneTo(spec.gnnGraph.num_nodes))]), false)
end

function GI.set_state!(game::GameEnv, state)
    game.gnnGraph = state.gnnGraph
    game.maskedActions = state.availableActions
    game.visitedVerticies = getPath(state.gnnGraph, game.visitedVerticies)
    any(game.maskedActions) || (game.finished = true)
    return
end

function getPath(gnnGraph, visitedVerticies)
    lastVertex = last(visitedVerticies)
    if isone(count(idx -> idx == lastVertex, gnnGraph.graph[1]))
        push!(visitedVerticies, gnnGraph.graph[2][findfirst(ind -> ind == lastVertex, gnnGraph.graph[1])])
    end
    # sources = gnnGraph.graph[1]
    # targets = gnnGraph.graph[2]
    # madeConnection(vertex) = isone(count(val -> val == vertex, sources))
    # isConnected(vertex) = isone(count(val -> val == vertex, targets))
    # startingVertex = filter(idx -> madeConnection(idx) && !isConnected(idx), unique(sources))
    # isnothing(startingVertex) && (startingVertex = 1)
    # path = [startingVertex]
    # foreach(path) do vertex
    #     if madeConnection(vertex)
    #         push!(path, targets[findfirst(idx -> idx == vertex, sources)])
    #     end
    # end
    return visitedVerticies
end
function Base.hash(gnn::GNNGraph, h::UInt64)
    hash(hash(gnn.graph[1]) + hash(gnn.graph[2]) + hash(gnn.graph[3]) + hash(gnn.ndata) + h)
end

GI.two_players(::GameSpec) = false
GI.actions(a::GameSpec) = collect(range(1, length = a.gnnGraph.num_nodes))
GI.clone(g::GameEnv) = GameEnv(g.gnnGraph, deepcopy(g.maskedActions), deepcopy(g.visitedVerticies), g.finished)
GI.white_playing(::GameEnv) = true
GI.game_terminated(g::GameEnv) = g.finished
GI.available_actions(g::GameEnv) = collect(range(1, length = g.gnnGraph.num_nodes))[g.maskedActions]
GI.actions_mask(game::GameEnv) = game.maskedActions

function GI.current_state(g::GameEnv)
    return (gnnGraph = g.gnnGraph, availableActions = g.maskedActions)
    # modifiedAdacencyList = deepcopy(g.graph[3])
    # if length(g.visitedVerticies) > 1
    #     foreach(enumerate(g.visitedVerticies[2:end])) do (idx, node)
    #         prevNode = state.path[idx]
    #         # Values are set to zero because SimpleWeightedGraphs discards edges with weights of zero.
    #         modifiedAdacencyList[prevNode, :] .= 0
    #         modifiedAdacencyList[:, node] .= 0
    #         modifiedAdacencyList[node, prevNode] = 0
    #         modifiedAdacencyList[prevNode, node] = spec.fadjlist[prevNode, node]
    #     end
    # end
    # return (ndata = (data = modifiedAdacencyList, x = ones(2)), availableActions = g.maskedActions)
end

function GI.play!(g::GameEnv, vertex::Int)
    sources = g.gnnGraph.graph[1]
    targets = g.gnnGraph.graph[2]
    weights = g.gnnGraph.graph[3]
    sourcesTargets = hcat(sources,targets)

    maskedActions = deepcopy(g.maskedActions)
    maskedActions[vertex] = false
    
    visitedVerticies = deepcopy(g.visitedVerticies)
    index = [last(visitedVerticies), vertex]

    potentialIndicies = findall(vert -> vert == index[1], sources)
    keptIndicie = findfirst(col -> col == index, collect(eachrow(sourcesTargets)))
    removedIndicies = filter(ind -> ind != keptIndicie, potentialIndicies)

    sourcesTargetsWeights = hcat(sourcesTargets, weights)
    newGraph = hcat(deleteat!(collect(eachrow(sourcesTargetsWeights)), removedIndicies)...)
    inputs = collect(eachrow(newGraph))
    graph = GNNGraph(Vector{Int}(inputs[1]), Vector{Int}(inputs[2]), Vector{Float32}(inputs[3]); ndata = (; x = ones(Float32, 1, g.gnnGraph.num_nodes)))

    state = (gnnGraph = graph, availableActions = maskedActions)
    GI.set_state!(g, state)
end

function GI.state_dim(game_spec::GameSpec)
    return size(game_spec.gnnGraph.ndata.x)[1]
end

function GI.white_reward(g::GameEnv)
    isempty(g.visitedVerticies[1:end-1]) && (return 0.0)
    sources = g.gnnGraph.graph[1]
    indicies = findall(idx -> idx ∈ g.visitedVerticies[1:end-1], sources)
    return -1 * sum(indicies) do vert
        g.gnnGraph.graph[3][vert]
    end
end

function GI.heuristic_value(g::GameEnv)
    return GI.white_reward(g)
end

# GI.render(g::GameEnv) = graphplot(g.gnnGraph.graph; curves = false)

function GI.graph_state(spec::GameSpec, state)
    return state.gnnGraph
end