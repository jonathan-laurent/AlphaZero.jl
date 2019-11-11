#####
##### Interface with the Flux.jl framework
#####

module FluxNets

export SimpleNet, SimpleNetHP, ResNet, ResNetHP

using ..Network
using Base: @kwdef
import ..GameInterface
import ..Util

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

using Flux: Tracker, relu, softmax
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection

#####
##### Flux Networks
#####

abstract type FluxNetwork{Game} <: AbstractNetwork{Game} end

function Base.copy(nn::Net) where Net <: FluxNetwork
  #new = Net(Network.hyperparams(nn))
  #Flux.loadparams!(new, Flux.params(nn))
  #return new
  return Base.deepcopy(nn)
end

Network.to_cpu(nn::FluxNetwork) = Flux.cpu(nn)

function Network.to_gpu(nn::FluxNetwork)
  CUARRAYS_IMPORTED && CuArrays.allowscalar(false)
  return Flux.gpu(nn)
end

Network.set_test_mode!(nn::FluxNetwork, mode) = Flux.testmode!(nn, mode)

Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Tracker.data(Flux.cpu(x))

Network.params(nn::FluxNetwork) = Flux.params(nn)

function Network.train!(nn::FluxNetwork, loss, data, lr)
  optimizer = Flux.ADAM(lr)
  Flux.train!(loss, Flux.params(nn), data, optimizer)
end

regularized_child_leaves(l) = []
regularized_child_leaves(l::Flux.Dense) = [l.W]
regularized_child_leaves(l::Flux.Conv) = [l.weight]

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

function gc(::FluxNetwork)
  CUARRAYS_IMPORTED || return
  GC.gc()
  CuArrays.clearpool()
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
##### Utilities for the networks library
#####

linearize(x) = reshape(x, :, size(x)[end])

#####
##### Include networks library
#####

include("SimpleNet.jl")
include("ResNet.jl")

end
