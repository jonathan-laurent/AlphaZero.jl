module Network


using CUDA
using Flux
using Random: MersenneTwister

using ..Util.Devices: Device, CPU, GPU, arr_is_on_gpu

export FluxNetwork, on_gpu, forward, to_cpu, to_gpu, set_test_mode!, SimpleNet, SimpleNetHP


"""
    FluxNetwork

Abstract type for neural networks implemented using the _Flux_ framework.

Provided that the above holds, `FluxNetwork` implements the full
network interface with the following exceptions:
[`Network.HyperParams`](@ref), [`Network.hyperparams`](@ref),
[`Network.forward`](@ref) and [`Network.on_gpu`](@ref).
"""
abstract type FluxNetwork end


#####
##### Interface
#####


"""
    HyperParams(::Type{<:FluxNetwork})

Return the hyperparameter type associated with a given network type.
"""
function HyperParams end


"""
    hyperparams(nn::FluxNetwork)

Return the hyperparameters of a network.
"""
function hyperparams end


"""
    on_gpu(nn::FluxNetwork) :: Bool

Test whether or not a network is located on GPU.
"""
function on_gpu end


"""
    forward(nn::FluxNetwork, states)

Compute the forward pass of a network on a batch of inputs.

Expect a `Float32` tensor `states` whose batch dimension is the last one.

Return a `(P, V)` triple where:

  - `P` is a matrix of size `(num_actions, batch_size)`. It is allowed
    to put weight on invalid actions (see [`evaluate`](@ref)).
  - `V` is a row vector of size `(1, batch_size)`
"""
function forward end


#####
##### Derived functions
#####


"""
    to_cpu(::FluxNetwork)

Return a copy of the given network that has been transferred to the CPU
or return the given network untouched if it is already on CPU.
"""
function to_cpu(nn::FluxNetwork)
    return Flux.cpu(nn)
end


"""
    to_gpu(::FluxNetwork)

Return a copy of the given network that has been transferred to the GPU
if one is available. Otherwise, return the given network untouched.
"""
function to_gpu(nn::FluxNetwork)
    CUDA.allowscalar(false)
    return Flux.gpu(nn)
end


"""
    set_test_mode!(mode)

Put a network in test mode or in training mode.
This is relevant for networks featuring layers such as
batch normalization layers.
"""
function set_test_mode!(nn::FluxNetwork, mode)
    Flux.testmode!(nn, mode)
end


"""
    num_parameters(nn::FluxNetwork)

Return the total number of parameters of a network.
"""
function num_parameters(nn::Net) where Net <: FluxNetwork
    return sum(length(p) for p in Flux.params(nn))
end


"""
    copy(nn:Net) where Net <:FluxNetwork

A copy function that also handles CPU/GPU transfers and
test/train mode switches.
"""
function Base.copy(nn::Net) where Net <: FluxNetwork
    return Base.deepcopy(nn)
end


"""
    gc(::FluxNetwork)

Perform full garbage collection and empty the GPU memory pool.
"""
function gc(::FluxNetwork)
    GC.gc(true)
end



#####
##### Include networks library
#####

include("SimpleNet.jl")
# include("architectures/resnet.jl")

end
