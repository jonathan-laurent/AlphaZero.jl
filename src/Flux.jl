#####
##### Interface with the Flux.jl framework
#####

using CUDAapi

# Import CuArrays only if CUDA is installed
if has_cuda()
  try
    using CuArrays
    @eval const CUARRAYS_IMPORTED = true
  catch ex
    @warn(
      "CUDA is installed, but CuArrays.jl fails to load.",
      exception=(ex,catch_backtrace()))
    @eval const CUARRAYS_IMPORTED = false
  end
else
  @eval const CUARRAYS_IMPORTED = false
end

import Flux
using Flux: Tracker, Chain, Dense, relu, softmax

if CUARRAYS_IMPORTED
  @eval begin
    CuArrays.allowscalar(false)
    on_gpu(::Type{<:Array}) = false
    on_gpu(::Type{<:CuArray}) = true
    on_gpu(::Type{<:Flux.TrackedArray{R,N,A}}) where {R, N, A} = on_gpu(A)
    on_gpu(x) = on_gpu(typeof(x))
  end
else
  @eval begin
    on_gpu(x) = false
  end
end

#####
##### Flux Networks
#####

abstract type FluxNetwork{Game} <: AbstractNetwork{Game} end

function Base.copy(nn::Net) where Net <: FluxNetwork
  new = Net(Network.hyperparams(nn))
  Flux.loadparams!(new, Flux.params(nn))
  return new
end

Network.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

Network.to_gpu(nn::FluxNetwork) = Flux.gpu(nn)

Network.set_test_mode!(nn::FluxNetwork, mode) = Flux.testmode!(nn, mode)

Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Tracker.data(Flux.cpu(x))

Network.num_parameters(nn::FluxNetwork) =
  sum(length(p) for p in Flux.params(nn))
  
function Network.train!(nn::FluxNetwork, loss, data, lr)
  optimizer = Flux.ADAM(lr)
  Flux.train!(loss, Flux.params(nn), data, optimizer)
end

regularized_child_leaves(l) = []
regularized_child_leaves(l::Flux.Dense) = [l.W]

# Inspired by the implementation of Flux.params
function Network.regularized_params(net::FluxNetwork)
  ps = Flux.Params()
  Flux.prefor(net) do p
    for r in regularized_child_leaves(p)
      any(x -> x === r, ps) || push!(ps, r)
    end
  end
  return ps
end

#####
##### Common functions between two-head neural networks
#####

abstract type TwoHeadNetwork{G} <: FluxNetwork{G} end

function Network.forward(nn::TwoHeadNetwork, board)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c)
  return (p, v)
end

# Flux.@treelike does not work do to Network being parametric
Flux.children(nn::TwoHeadNetwork) = (nn.common, nn.vbranch, nn.pbranch)

function Flux.mapchildren(f, nn::Net) where Net <: TwoHeadNetwork
  Net(nn.hyper, f(nn.common), f(nn.vbranch), f(nn.pbranch))
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.on_gpu(nn::TwoHeadNetwork) = on_gpu(nn.vbranch[end].b)

#####
##### Simple Example Network
#####

@kwdef struct SimpleNetHyperParams
  width :: Int
  depth_common :: Int
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
end

struct SimpleNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function linearize(x)
  s = size(x)
  return reshape(x, prod(s[1:end-1]), s[end])
end

function SimpleNet{G}(hyper::SimpleNetHyperParams) where G
  indim = GI.board_dim(G)
  outdim = GI.num_actions(G)
  hsize = hyper.width
  hlayers(depth) = [Dense(hsize, hsize, relu) for i in 1:depth]
  common = Chain(
    linearize,
    Dense(indim, hsize, relu),
    hlayers(hyper.depth_common)...)
  vbranch = Chain(
    hlayers(hyper.depth_vbranch)...,
    Dense(hsize, 1, tanh))
  pbranch = Chain(
    hlayers(hyper.depth_pbranch)...,
    Dense(hsize, outdim),
    softmax)
  SimpleNet{G}(hyper, common, vbranch, pbranch)
end

Network.HyperParams(::Type{<:SimpleNet}) = SimpleNetHyperParams

#####
##### Dense Resnet
#####

@kwdef struct ResNetHyperParams
  width :: Int
  num_blocks :: Int = 5
end

struct ResNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function ResNetBlock(dim, hdim)
  layers = Flux.Chain(
    Flux.Dense(dim, hdim),
    Flux.BatchNorm(hdim, relu),
    Flux.Dense(hdim, dim),
    Flux.BatchNorm(dim))
  return Flux.Chain(
    Flux.SkipConnection(layers, +),
    Flux.relu)
end

function NormalizedDenseLayer(indim, outdim)
  return Flux.Chain(
    Flux.Dense(indim, outdim),
    Flux.BatchNorm(outdim, relu))
end

function ResNet{G}(hyper::ResNetHyperParams) where G
  indim = GI.board_dim(G)
  outdim = GI.num_actions(G)
  hsize = hyper.width
  common = Flux.Chain(
    NormalizedDenseLayer(indim, hsize),
    [ResNetBlock(hsize, hsize) for i in 1:hyper.num_blocks]...)
  vbranch = Flux.Chain(
    NormalizedDenseLayer(hsize, hsize),
    Dense(hsize, 1, tanh))
  pbranch = Flux.Chain(
    NormalizedDenseLayer(hsize, hsize),
    Dense(hsize, outdim),
    softmax)
  ResNet{G}(hyper, common, vbranch, pbranch)
end

Network.HyperParams(::Type{<:ResNet}) = ResNetHyperParams
