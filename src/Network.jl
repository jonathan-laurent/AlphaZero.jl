"""
A generic, framework agnostic interface for neural networks.
"""
module Network

export AbstractNetwork

import ..MCTS
import ..GameInterface

using ..Util: @unimplemented

"""
    AbstractNetwork{Game} <: MCTS.Oracle{Game}

Abstract base type for a neural network.
  
---
  
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
    forward(::AbstractNetwork, board)
    
Compute the forward pass of the network on a batch of inputs
(the batch dimension being the last one).

Return a `(P, V)` triple. The probability vector `P` is allowed to put
some weight on disallowed actions.
"""
function forward(::AbstractNetwork, board)
  @unimplemented
end

"""
    train!(::AbstractNetwork, loss, data, learning_rate)
    
Train a given network on data.
"""
function train!(::AbstractNetwork, loss, data, learning_rate)
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
    num_parameters(::AbstractNetwork)

Return the total number of parameters of a network.
"""
function num_parameters(::AbstractNetwork)
  @unimplemented
end

#####
##### Derived functions
#####

function evaluate(nn::AbstractNetwork, board, actions_mask)
  p, v = forward(nn, board)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p  = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1 .- sp
  return (p, v, p_invalid)
end

to_singleton_batch(x) = reshape(x, size(x)..., 1)
from_singleton_batch(x) = reshape(x, size(x)[1:end-1])

function MCTS.evaluate(nn::AbstractNetwork{G}, board, available_actions) where G
  x = GameInterface.vectorize_board(G, board)
  a = GameInterface.actions_mask(G, available_actions)
  xnet, anet = to_singleton_batch.(convert_input_tuple(nn, (x, a)))
  p, v, _ = from_singleton_batch.(
    convert_output_tuple(nn, evaluate(nn, xnet, anet)))
  return (p[a], v[1])
end

function MCTS.evaluate_batch(nn::AbstractNetwork{G}, batch) where G
  X = Util.concat_columns((
    GameInterface.vectorize_board(G, b)
    for (b, as) in batch))
  A = Util.concat_columns((
    GameInterface.actions_mask(G, as)
    for (b, as) in batch))
  Xnet, Anet = convert_input_tuple(nn, (X, A))
  P, V, _ = convert_output_tuple(nn, evaluate(nn, Xnet, Anet))
  return [(P[A[:,i],i], V[1,i]) for i in eachindex(batch)]
end

function copy(network; on_gpu, test_mode)
  network = Base.copy(network)
  network = on_gpu ? to_gpu(network) : to_cpu(network)
  set_test_mode!(network, test_mode)
  return network
end

end
