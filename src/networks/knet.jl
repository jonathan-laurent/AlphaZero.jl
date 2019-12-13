#####
##### Interface with the Knet.jl framework
#####

module KNets

export SimpleNet, SimpleNetHP, ResNet, ResNetHP

using ..Network
using Base: @kwdef
import ..GameInterface
import ..Util

import Knet

include("knet/layers.jl")

#####
##### Traversal utilities
#####

children(x) = ()

mapchildren(f, x) = Util.@unimplemented

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

abstract type KNetwork{Game} <: AbstractNetwork{Game} end

Base.copy(nn::KNetwork) = Base.deepcopy(nn)

const GPU_AVAILABLE = Knet.gpu() >= 0

Network.to_gpu(nn::KNetwork) = GPU_AVAILABLE ? Knet.gpucopy(nn) : nn
Network.to_cpu(nn::KNetwork) = GPU_AVAILABLE ? Knet.cpucopy(nn) : nn

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

function Network.train!(nn::KNetwork, loss, data, lr)
  Knet.adam(loss, data, lr=lr) |> collect
end

function Network.gc(::KNetwork)
  GPU_AVAILABLE || return
  GC.gc(true)
  Knet.gc()
end

#####
##### Common functions between two-head neural networks
#####

abstract type TwoHeadNetwork{G} <: KNetwork{G} end

function Network.forward(nn::TwoHeadNetwork, board)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c)
  return (p, v)
end

children(nn::TwoHeadNetwork) = (nn.common, nn.vbranch, nn.pbranch)

function mapchildren(f, nn::Net) where Net <: TwoHeadNetwork
  Net(nn.hyper, f(nn.common), f(nn.vbranch), f(nn.pbranch))
end

Network.hyperparams(nn::TwoHeadNetwork) = nn.hyper

function Network.on_gpu(nn::TwoHeadNetwork)
  b = nn.vbranch.layers[end].b
  return isa(Knet.value(b), Knet.KnetArray)
end

#####
##### Include networks library
#####

include("architectures/simplenet.jl")
include("architectures/resnet.jl")

end
