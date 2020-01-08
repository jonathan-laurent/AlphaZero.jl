"""
This module provides utilities to build neural networks with Flux,
along with a library of standard architectures.
"""
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
    @show CuArrays.usage_limit[]
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
    on_gpu(x) = on_gpu(typeof(x))
  end
else
  @eval begin
    on_gpu(x) = false
  end
end

using Flux: relu, softmax
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

# TODO: dirty hack to cope with the fact that Flux 0.10 removed Flux.testmode!
const TEST_MODE = IdDict()

function Network.set_test_mode!(nn::FluxNetwork, mode)
  TEST_MODE[nn] = mode
end

Network.convert_input(nn::FluxNetwork, x) =
  Network.on_gpu(nn) ? Flux.gpu(x) : x

Network.convert_output(nn::FluxNetwork, x) = Flux.cpu(x)

Network.params(nn::FluxNetwork) = Flux.params(nn)

function Network.train!(nn::FluxNetwork, loss, data, lr)
  optimizer = Flux.ADAM(lr)
  Flux.train!(loss, Flux.params(nn), data, optimizer)
end

regularized_child_leaves(l) = []
regularized_child_leaves(l::Flux.Dense) = [l.W]
regularized_child_leaves(l::Flux.Conv) = [l.weight]

# Reimplementation of what used to be Flux.prefor, does not visit leafs
function foreach_flux_node(f::Function, x, seen = IdDict())
  Flux.isleaf(x) && return
  haskey(seen, x) && return
  seen[x] = true
  f(x)
  for child in Flux.trainable(x)
    foreach_flux_node(f, child, seen)
  end
end

function Network.regularized_params(net::FluxNetwork)
  ps = Flux.Params()
  foreach_flux_node(net) do p
    for r in regularized_child_leaves(p)
      any(x -> x === r, ps) || push!(ps, r)
    end
  end
  return ps
end

function Network.gc(::FluxNetwork)
  CUARRAYS_IMPORTED || return
  GC.gc(true)
  CuArrays.reclaim()
end

#####
##### Common functions between two-head neural networks
#####

abstract type TwoHeadNetwork{G} <: FluxNetwork{G} end

function Network.forward(nn::TwoHeadNetwork, board)
  # TODO: eliminate this horror when Flux catches up
  # @eval Flux.istraining() = $(!get(TEST_MODE, nn, false))
  c = nn.common(board)
  v = nn.vhead(c)
  p = nn.phead(c)
  return (p, v)
end

# Flux.@functor does not work do to Network being parametric
function Flux.functor(nn::Net) where Net <: TwoHeadNetwork
  children = (nn.common, nn.vhead, nn.phead)
  constructor = cs -> Net(nn.hyper, cs...)
  return (children, constructor)
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.on_gpu(nn::TwoHeadNetwork) = on_gpu(nn.vhead[end].b)

#####
##### Utilities for the networks library
#####

linearize(x) = reshape(x, :, size(x)[end])

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")

end
