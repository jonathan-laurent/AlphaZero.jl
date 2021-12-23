using GraphNeuralNetworks
using Statistics

"""
Super, super demo example

network will be this 
nodeFeature -> number of features on node
innerSize -> how large the inner network layer will be
actionCount -> how many actions plus 1 (network importance)

GNNChain(GCNConv(nodeFeature => innerSize),
                        BatchNorm(innerSize),     # Apply batch normalization on node features (nodes dimension is batch dimension)
                        x -> relu.(x),     
                        GCNConv(64 => innerSize, relu),
                        GlobalPool(mean),  # aggregate node-wise features into graph-wise features
                        Dense(innerSize, actionCount+1)) |> device;
"""
@kwdef struct SimpleGraphNetHP
  innerSize :: Int
end

"""
    SimpleGNN <: FluxGNN

    Something simple
"""
mutable struct SimpleGNN <: FluxGNN
  gspec
  hyper
  model
end


function SimpleGNN(gspec::AbstractGameSpec, hyper::SimpleGraphNetHP)
    innerSize = hyper.innerSize
    nodeFeature = GI.state_dim(gspec)
    actionCount = GI.num_actions(gspec)+1
    model = GNNChain(GCNConv(nodeFeature[1] => innerSize),
        BatchNorm(innerSize),     # Apply batch normalization on node features (nodes dimension is batch dimension)
        x -> relu.(x),     
        GCNConv(innerSize => innerSize, relu),
        GlobalPool(mean),
        Dense(innerSize, actionCount),
        softmax)
    SimpleGNN(gspec, hyper, model)
end

Network.HyperParams(::Type{<:SimpleGNN}) = SimpleGraphNetHP

function Base.copy(nn::SimpleGNN)
  return SimpleGNN(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.model)
  )
end
result(graph, model) = model(graph, g.ndata.x)

function Network.forward(nn::SimpleGNN, state)
    applyModel(graph) = nn.model(graph, graph.ndata.x)
    result = applyModel.(state)
    # Matrix{Float32}(undef, GI.num_actions(nn.gspec)+1, length(state))
    # for (ind, graph) in enumerate(state)
    #     result[:, ind] .= nn.model(graph, graph.ndata.x)
    # end
    v = [result[ind][indDepth] for indDepth in 1:1, ind in 1:length(state)]
    p = [result[ind][indDepth] for indDepth in 2:size(result[1], 1), ind in 1:length(state) ]
    return (p, v)
end