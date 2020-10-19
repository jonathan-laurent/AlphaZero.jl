"""
This module provides utilities to build neural networks with Knet,
along with a library of standard architectures.
"""
module KnetLib

export SimpleNet, SimpleNetHP, ResNet, ResNetHP

using ..AlphaZero

using Base: @kwdef

import CUDA
import Knet

include("knet/layers.jl")

#####
##### Traversal utilities
#####

children(x) = ()

# mapchildren(f, x) = x'
function mapchildren end

function traverse!(f, model, seen=IdDict())
  haskey(seen, model) && return
  seen[model] = true
  f(model)
  for c in children(model)
    traverse!(f, c, seen)
  end
end

function gather(f, model)
  acc = Any[]
  traverse!(x -> append!(acc, f(x)), model)
  return acc
end

function fmap(f, model, cache=IdDict())
  haskey(cache, model) && (return cache[model])
  return cache[model] =
    isempty(children(model)) ?
      f(model) :
      f(mapchildren(c -> fmap(f, c, cache), model))
end

#####
##### Implementing the `Network` interface
#####

"""
    KNetwork <: AbstractNetwork

Abstract type for neural networks implemented using the _Knet_ framework.

  - Subtypes are expected to be expressed as the composition of Flux-like
    layers that implement a functor interface through functions `children` and
    `mapchildren`.
  - A custom implementation of `regularized_params_` must also
    be implemented for layers containing parameters that are subject to
    regularization.

Provided that the above holds, `KNetwork` implements the full network interface
with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type KNetwork <: AbstractNetwork end

Base.copy(nn::KNetwork) = Base.deepcopy(nn)

Network.to_gpu(nn::KNetwork) = CUDA.functional() ? Knet.gpucopy(nn) : nn
Network.to_cpu(nn::KNetwork) = CUDA.functional() ? Knet.cpucopy(nn) : nn

params_(x) = []
params_(x::Knet.Param) = [x]
Network.params(nn::KNetwork) = gather(params_, nn)

regularized_params_(x) = []
regularized_params_(m::Dense) = [m.W]
regularized_params_(m::Conv) = [m.W]
Network.regularized_params(nn::KNetwork) = gather(regularized_params_, nn)

set_test_mode!_(x, mode) = []
set_test_mode!_(l::BatchNorm, mode) = l.train = !mode

Network.set_test_mode!(nn::KNetwork, mode) =
  traverse!(x -> set_test_mode!_(x, mode), nn)

Network.convert_input(nn::KNetwork, x) =
  Network.on_gpu(nn) ? Knet.KnetArray(x) : Array(x)

Network.convert_output(::KNetwork, x) = x
Network.convert_output(::KNetwork, x::Knet.KnetArray) = Array(x)

function Network.train!(
    callback, nn::KNetwork, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  optimiser = Knet.Nesterov(lr=opt.lr_low, gamma=opt.momentum_high)
  for (i, l) in enumerate(Knet.minimize(loss, data, optimiser))
    callback(i, l)
    optimiser.lr = lr[i]
    optimiser.gamma = momentum[i]
  end
end

function Network.train!(callback, nn::KNetwork, opt::Adam, loss, data, n)
  optimiser = Knet.Adam(lr=opt.lr)
  for (i, l) in enumerate(Knet.minimize(loss, data, optimiser))
    callback(i, l)
  end
end

function Network.gc(::KNetwork)
  CUDA.functional() || return
  GC.gc(true)
end

#####
##### Common functions between two-head neural networks
#####

"""
    TwoHeadNetwork <: KNetwork

An abstract type for two-head neural networks implemented with Knet.

Subtypes are assumed to have fields
`hyper`, `gspec`, `common`, `vhead` and `phead`. Based on those, an implementation
is provided for [`Network.hyperparams`](@ref), [`Network.game_spec`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref), leaving only
[`Network.HyperParams`](@ref) to be implemented.
"""
abstract type TwoHeadNetwork <: KNetwork end

function Network.forward(nn::TwoHeadNetwork, state)
  c = nn.common(state)
  v = nn.vhead(c)
  p = nn.phead(c)
  return (p, v)
end

children(nn::TwoHeadNetwork) = (nn.common, nn.vhead, nn.phead)

function mapchildren(f, nn::Net) where Net <: TwoHeadNetwork
  Net(nn.gspec, nn.hyper, f(nn.common), f(nn.vhead), f(nn.phead))
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

Network.game_spec(nn::TwoHeadNetwork) = nn.gspec

function Network.on_gpu(nn::TwoHeadNetwork)
  b = nn.vhead.layers[end].b
  return isa(Knet.value(b), Knet.KnetArray)
end

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")

end
