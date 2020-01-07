"""
A generic, framework agnostic interface for neural networks.
"""
module Network

export AbstractNetwork, OptimiserSpec, Momentum, CyclicMomentum

import ..MCTS, ..GameInterface, ..Util

using Base: @kwdef
using ..Util: @unimplemented
using Statistics: mean

"""
    AbstractNetwork{Game} <: MCTS.Oracle{Game}

Abstract base type for a neural network.

# Constructor

Any subtype `Network` must implement the following constructor:

    Network(hyperparams)

where the expected type of `hyperparams` is given by
[`HyperParams(Network)`](@ref HyperParams).
"""
abstract type AbstractNetwork{G} <: MCTS.Oracle{G} end

#####
##### Interface
#####

"""
    HyperParams(::Type{<:AbstractNetwork})

Return the hyperparameter type associated with a given network type.
"""
function HyperParams(::Type{<:AbstractNetwork})
  @unimplemented
end

"""
    hyperparams(::AbstractNetwork)

Return the hyperparameters of a network.
"""
function hyperparams(::AbstractNetwork)
  @unimplemented
end

"""
    Base.copy(::AbstractNetwork)

Return a copy of the given network.
"""
function Base.copy(::AbstractNetwork)
  @unimplemented
end

"""
    to_gpu(::AbstractNetwork)

Return a copy of the given network that has been transferred to the GPU
if one is available. Otherwise, return the given network untouched.
"""
function to_gpu(::AbstractNetwork)
  @unimplemented
end

"""
    to_cpu(::AbstractNetwork)

Return a copy of the given network that has been transferred to the CPU
or return the given network untouched if it is already on CPU.
"""
function to_cpu(::AbstractNetwork)
  @unimplemented
end

"""
    on_gpu(::AbstractNetwork)

Test if a network is located on GPU.
"""
function on_gpu(::AbstractNetwork)
  @unimplemented
end

"""
    set_test_mode!(mode=true)

Put a network in test mode or in training mode.
This is relevant for networks featuring layers such as
batch normalization layers.
"""
function set_test_mode!(mode=true)
  @unimplemented
end

"""
    convert_input(::AbstractNetwork, input)

Convert an array (or number) to the right format so that it can be used
as an input by a given network.
"""
function convert_input(::AbstractNetwork, input)
  @unimplemented
end

function convert_input_tuple(nn::AbstractNetwork, input::Tuple)
  return map(input) do arr
    convert_input(nn, arr)
  end
end

"""
    convert_output(::AbstractNetwork, output)

Convert an array (or number) produced by a neural network
to a standard CPU array (or number) type.
"""
function convert_output(::AbstractNetwork, output)
  @unimplemented
end

function convert_output_tuple(nn::AbstractNetwork, output::Tuple)
  return map(output) do arr
    convert_output(nn, arr)
  end
end

"""
    forward(::AbstractNetwork, boards)

Compute the forward pass of the network on a batch of inputs.

Expect a `Float32` tensor `boards` whose batch dimension is the last one.

Return a `(P, V)` triple where:

  - `P` is a matrix of size `(num_actions, batch_size)`. It is allowed
    to put weight on invalid actions (see [`evaluate`](@ref)).
  - `V` is a row vector of size `(1, batch_size)`
"""
function forward(::AbstractNetwork, boards)
  @unimplemented
end

"""
    regularized_params(::AbstractNetwork)

Return the collection of regularized parameters of a network.
This usually excludes neuron's biases.
"""
function regularized_params(::AbstractNetwork)
  @unimplemented
end

"""
    params(::AbstractNetwork)

Return the collection of trainable parameters of the network.
"""
function params(::AbstractNetwork)
  @unimplemented
end

"""
    gc(::AbstractNetwork)

Perform full garbage collection.
"""
function gc(::AbstractNetwork)
  @unimplemented
end

# Optimizers and training

abstract type OptimiserSpec end

@kwdef struct Momentum <: OptimiserSpec
  lr :: Float32
  momentum :: Float32
end

@kwdef struct CyclicMomentum <: OptimiserSpec
  lr_low :: Float32
  lr_high :: Float32
  momentum_low :: Float32
  momentum_high :: Float32
end

"""
    train!(::AbstractNetwork, opt::OptimiserSpec, loss, data, learning_rate)

Train a given network on data.
"""
function train!(::AbstractNetwork, opt::OptimiserSpec, loss, data)
  @unimplemented
end

#####
##### Derived functions
#####

"""
    num_parameters(::AbstractNetwork)

Return the total number of parameters of a network.
"""
function num_parameters(nn::AbstractNetwork)
  return sum(length(p) for p in params(nn))
end

"""
    num_regularized_parameters(::AbstractNetwork)

Return the total number of regularized parameters of a network.
"""
function num_regularized_parameters(nn::AbstractNetwork)
  return sum(length(p) for p in regularized_params(nn))
end

"""
    mean_weight(::AbstractNetwork)

Return the mean absolute value of the regularized parameters of a network.
"""
function mean_weight(nn::AbstractNetwork)
  sw = sum(sum(abs.(p)) for p in regularized_params(nn))
  sw = convert_output(nn, sw)
  return sw / num_regularized_parameters(nn)
end

"""
    evaluate(network::AbstractNetwork, boards, action_masks)

Evaluate a batch of board positions. This function is a wrapper
on [`forward`](@ref) that puts a zero weight on invalid actions.

# Arguments

  - `boards` is a tensor whose last dimension has size `bach_size`
  - `action_masks` is a binary matrix of size `(num_actions, batch_size)`

# Return

Return a `(P, V, Pinv)` triple where:

  - `P` is a matrix of size `(num_actions, batch_size)`.
  - `V` is a row vector of size `(1, batch_size)`.
  - `Pinv` is a row vector of size `(1, batch_size)`
     that indicates the total probability weight put by the network
     on invalid actions for each sample.

All tensors manipulated by this function have type `Float32`.
"""
function evaluate(nn::AbstractNetwork, board, actions_mask)
  p, v = forward(nn, board)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1 .- sp
  return (p, v, p_invalid)
end

to_singleton_batch(x) = reshape(x, size(x)..., 1)
from_singleton_batch(x) = reshape(x, size(x)[1:end-1])

"""
    MCTS.evaluate(::AbstractNetwork, board, available_actions)

Evaluate a single board position.
See [`MCTS.evaluate(::MCTS.Oracle)`](@ref).

!!! warning
    Evaluating a neural network on a single sample at a time is slow.
    When possible, use [`MCTS.evaluate_batch`](@ref
    MCTS.evaluate_batch(::AbstractNetwork)) instead.
"""
function MCTS.evaluate(nn::AbstractNetwork{G}, board, available_actions) where G
  x = GameInterface.vectorize_board(G, board)
  a = GameInterface.actions_mask(G, available_actions)
  xnet, anet = to_singleton_batch.(convert_input_tuple(nn, (x, Float32.(a))))
  p, v, _ = from_singleton_batch.(
    convert_output_tuple(nn, evaluate(nn, xnet, anet)))
  return (p[a], v[1])
end

"""
    MCTS.evaluate_batch(::AbstractNetwork, batch)

Evaluate a batch of positions.
See [`MCTS.evaluate_batch(::MCTS.Oracle)`](@ref).
"""
function MCTS.evaluate_batch(nn::AbstractNetwork{G}, batch) where G
  X = Util.superpose((
    GameInterface.vectorize_board(G, b)
    for (b, as) in batch))
  A = Util.superpose((
    GameInterface.actions_mask(G, as)
    for (b, as) in batch))
  Xnet, Anet = convert_input_tuple(nn, (X, Float32.(A)))
  P, V, _ = convert_output_tuple(nn, evaluate(nn, Xnet, Anet))
  return [(P[A[:,i],i], V[1,i]) for i in eachindex(batch)]
end

"""
    copy(::AbstractNetwork; on_gpu, test_mode)

A copy function that also handles CPU/GPU transfers and
test/train mode switches.
"""
function copy(network::AbstractNetwork; on_gpu, test_mode)
  network = Base.copy(network)
  network = on_gpu ? to_gpu(network) : to_cpu(network)
  set_test_mode!(network, test_mode)
  return network
end

end
