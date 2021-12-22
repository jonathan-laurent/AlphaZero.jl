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
    nodeFeature = GI.state_dim(gspec; graph = true)
    actionCount = length(GI.num_actions(gspec))
    model = GNNChain(GCNConv(nodeFeature[1] => innerSize),
        BatchNorm(innerSize),     # Apply batch normalization on node features (nodes dimension is batch dimension)
        x -> relu.(x),     
        GCNConv(innerSize => innerSize, relu),
        GlobalPool(mean),  # aggregate node-wise features into graph-wise features
        Dense(innerSize, actionCount))
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

function Network.forward(nn::SimpleGNN, state)
  c = nn.model(state, state.ndata.x)
  v = c[1] # Value of state
  p = c[2:end] # Ranking of actions
  return (p, v)
end