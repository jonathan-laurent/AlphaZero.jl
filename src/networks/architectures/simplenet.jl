"""
    SimpleNetHP

Hyperparameters for the simplenet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Number of neurons on each dense layer        |
| `depth_common :: Int`         | Number of dense layers in the trunk          |
| `depth_phead = 1`             | Number of hidden layers in the actions head  |
| `depth_vhead = 1`             | Number of hidden layers in the value  head   |
| `use_batch_norm = false`      | Use batch normalization between each layer   |
| `batch_norm_momentum = 0.6f0` | Momentum of batch norm statistics updates    |
"""
@kwdef struct SimpleNetHP
  width :: Int
  depth_common :: Int
  depth_phead :: Int = 1
  depth_vhead :: Int = 1
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
end

"""
    SimpleNet <: TwoHeadNetwork

A simple two-headed architecture with only dense layers.
"""
mutable struct SimpleNet <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function SimpleNet(gspec::AbstractGameSpec, hyper::SimpleNetHP)
  bnmom = hyper.batch_norm_momentum
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      Dense(indim, outdim, relu)
    end
  end
  indim = prod(GI.state_dim(gspec))
  outdim = GI.num_actions(gspec)
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  common = Chain(
    flatten,
    make_dense(indim, hsize),
    hlayers(hyper.depth_common)...)
  vhead = Chain(
    hlayers(hyper.depth_vhead)...,
    Dense(hsize, 1, tanh))
  phead = Chain(
    hlayers(hyper.depth_phead)...,
    Dense(hsize, outdim),
    softmax)
  SimpleNet(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{SimpleNet}) = SimpleNetHP

function Base.copy(nn::SimpleNet)
  return SimpleNet(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end