"""
    Network

The `Network` module defines an abstract type `FluxNetwork` that serves as an interface for
neural networks built using the Flux framework. This interface allows for easy integration
with other modules requiring neural network functionality.

## Abstract Types

- [`FluxNetwork`](@ref): Abstract type for neural networks. Implements most of the network
  interface except for certain functions like `HyperParams`, `hyperparams`, `on_gpu`, and
  `forward`.

## Interface Functions

- [`HyperParams(::Type{<:FluxNetwork})`](@ref): Returns the hyperparameter type associated
    with a given network type.
- [`hyperparams(nn::FluxNetwork)`](@ref): Returns the hyperparameters of a network.
- [`on_gpu(nn::FluxNetwork) :: Bool`](@ref): Checks whether the network is on the GPU.
- [`forward(nn::FluxNetwork, states, use_softmax=false)`](@ref): Performs a forward pass of
    the network on a batch of inputs.

## Derived Functions

- [`to_cpu(::FluxNetwork)`](@ref): Transfers the network to the CPU.
- [`to_gpu(::FluxNetwork)`](@ref): Transfers the network to the GPU if available.
- [`set_train_mode!(nn::FluxNetwork)`](@ref): Puts the network in training mode.
- [`set_test_mode!(nn::FluxNetwork)`](@ref): Puts the network in test mode.
- [`num_parameters(nn::FluxNetwork)`](@ref): Returns the total number of parameters in
    the network.
- [`copy(nn::FluxNetwork)`](@ref): Copies the network.
- [`gc(::FluxNetwork)`](@ref): Empties the GPU memory pool and performs full garbage
    collection.

## Included Libraries

- SimpleNet.jl: Provides the implementation of Multi-Layer Perceptrons.
- SimpleResNet.jl: Provides the implementation of Multi-Layer Perceptrons with Residual
    blocks.

## Example

```julia
# Create a network of a certain type
hp = SimpleNetHP(...)  # replace ... with actual parameters
nn = SimpleNet(input_size, output_size, hp)

# Perform a forward pass
v, p = forward(nn, states)
```
"""
module Network

using CUDA
using Flux
using Random: MersenneTwister

using ..Util.Devices: Device, CPU, GPU, arr_is_on_gpu

export FluxNetwork
export SimpleNet, SimpleNetHP
export SimpleResNet, SimpleResNetHP
export on_gpu, forward, to_cpu, to_gpu, set_train_mode!, set_test_mode!


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
    forward(nn::FluxNetwork, states, use_softmax=false)

Compute the forward pass of a network on a batch of inputs.

Expect a `Float32` tensor `states` whose batch dimension is the last one.

Return a `(v, p)` tuple where:

- `v` is a row vector of size `(1, batch_size)` containing the value
    function estimates for each state in the batch.
- `p` is a matrix of size `(num_actions, batch_size)` containing the logits/policy
    (depending on `use_softmax`) for each state in the batch.
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
    return Flux.gpu(nn)
end


"""
    set_train_mode!(nn::Net) where Net <: FluxNetwork

Put a network in train mode. This is relevant for networks
featuring layers such as batch normalization layers.
"""
function set_train_mode!(nn::FluxNetwork)
    Flux.trainmode!(nn)
end


"""
    set_test_mode!(nn::Net) where Net <: FluxNetwork

Put a network in test mode. This is relevant for networks
featuring layers such as batch normalization layers.
"""
function set_test_mode!(nn::FluxNetwork)
    Flux.testmode!(nn)
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
include("SimpleResNet.jl")

end
