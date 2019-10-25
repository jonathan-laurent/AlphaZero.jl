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
  end
else
  @eval begin
    on_gpu(::Type) = false
  end
end
on_gpu(x) = on_gpu(typeof(x))

#####
##### Flux Networks
#####

abstract type FluxNetwork{Game} <: Network{Game} end

function Base.copy(nn::Net) where Net <: FluxNetwork
  new = Net(Networks.hyperparams(nn))
  Flux.loadparams!(new, Flux.params(nn))
  return new
end

Networks.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

Networks.to_gpu(nn::FluxNetwork) = Flux.gpu(nn)

function Networks.convert_input(nn::FluxNetwork, x)
  return Networks.on_gpu(nn) ? Flux.gpu(x) : x
end

function Networks.convert_output(nn::FluxNetwork, x)
  return Tracker.data(Flux.cpu(x))
end

function Networks.train!(nn::FluxNetwork, loss, data, lr)
  optimizer = Flux.ADAM(lr)
  Flux.train!(loss, Flux.params(nn), data, optimizer)
end

#####
##### Simple Example Network
#####

@kwdef struct SimpleNetHyperParams
  width :: Int = 300
  depth_common :: Int = 3
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
end

struct SimpleNet{Game} <: FluxNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function SimpleNet{G}(hyper::SimpleNetHyperParams) where G
  indim = GI.board_dim(G)
  outdim = GI.num_actions(G)
  hsize = hyper.width
  hlayers(depth) = [Dense(hsize, hsize, relu) for i in 1:depth]
  common = Chain(
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

Networks.HyperParams(::Type{<:SimpleNet}) = SimpleNetHyperParams

Networks.hyperparams(nn::SimpleNet) = nn.hyper

function Networks.forward(nn::SimpleNet, board)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c)
  return (p, v)
end

# Flux.@treelike does not work do to Network being parametric
Flux.children(nn::SimpleNet) = (nn.common, nn.vbranch, nn.pbranch)

function Flux.mapchildren(f, nn::Net) where Net <: SimpleNet
  Net(nn.hyper, f(nn.common), f(nn.vbranch), f(nn.pbranch))
end

function Networks.regularized_weights(nn::SimpleNet)
  W(mlp) = [l.W for l in mlp if isa(l, Dense)]
  return [W(nn.common); W(nn.vbranch); W(nn.pbranch)]
end

function Networks.network_report(nn::SimpleNet) :: Report.Network
  Ws = Tracker.data.(Networks.regularized_weights(nn))
  maxw = maximum(maximum(abs.(W)) for W in Ws)
  meanw = mean(mean(abs.(W)) for W in Ws)
  pbiases = nn.pbranch[end-1].b |> Tracker.data
  vbias = nn.vbranch[end].b |> Tracker.data
  return Report.Network(maxw, meanw, pbiases, sum(vbias))
end

Networks.num_parameters(nn::SimpleNet) = sum(length(p) for p in Flux.params(nn))

Networks.on_gpu(nn::SimpleNet) = on_gpu(nn.vbranch[end].b)

#####
##### Dense Resnet
#####
