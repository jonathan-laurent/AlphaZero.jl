"""
A generic, framework agnostic interface for neural networks.
"""
module Networks

export Network

import ..MCTS
import ..GameInterface
import ..Report

using ..Util: @unimplemented

"""
    Network{Game} <: MCTS.Oracle{Game}

Abstract base type for a neural network.
  
---
  
Any subtype `MyNet` must implement the following constructor:

    MyNet(hyperparams)
    
where the expected type of `hyperparams` is given by
[`HyperParams(MyNet)`](@ref HyperParams).
"""
abstract type Network{G} <: MCTS.Oracle{G} end

#####
##### Interface
#####

"""
    HyperParams(::Type{<:Network})
    
Return the hyperparameter type associated with a given network type.
"""
function HyperParams(::Type{<:Network})
  @unimplemented
end

"""
    hyperparams(::Network)
    
Return the hyperparameters of a network.
"""
function hyperparams(::Network)
  @unimplemented
end

"""
    Base.copy(::Network)
    
Return a copy of the given network.
"""
function Base.copy(::Network)
  @unimplemented
end

"""
    to_gpu(::Network)
    
Return a copy of the given network that has been transferred to the GPU
if one is available. Otherwise, return the given network untouched.
"""
function to_gpu(::Network)
  @unimplemented
end

"""
    to_cpu(::Network)
    
Return a copy of the given network that has been transferred to the CPU
or return the given network untouched if it is already on CPU.
"""
function to_cpu(::Network)
  @unimplemented
end

"""
    on_gpu(::Network)
    
Test if a network is located on GPU.
"""
function on_gpu(::Network)
  @unimplemented
end

"""
    convert_input(::Network, input)
    
Convert an array (or number) to the right format so that it can be used
as an input by a given network.
"""
function convert_input(::Network, input)
  @unimplemented
end

function convert_input_tuple(nn::Network, input::Tuple)
  return map(input) do arr
    convert_input(nn, arr)
  end
end

"""
    convert_output(::Network, output)
    
Convert an array (or number) produced by a neural network
to a standard CPU array (or number) type.
"""
function convert_output(::Network, output)
  @unimplemented
end

function convert_output_tuple(nn::Network, output::Tuple)
  return map(output) do arr
    convert_output(nn, arr)
  end
end

"""
    forward(::Network, board)
    
Compute the forward pass of the network on a single input
or on a batch of inputs (in which case the batch dimension is the last one).

Return a `(P, V)` triple. The probability vector `P` is allowed to put
some weight on disallowed actions.
"""
function forward(::Network, board)
  @unimplemented
end

"""
    train!(::Network, loss, data, learning_rate)
    
Train a given network on data.
"""
function train!(::Network, loss, data, learning_rate)
  @unimplemented
end

"""
    regularized_weights(::Network)
    
Return the collection of regularized weights of a network.
This usually excludes neuron's biases.
"""
function regularized_weights(::Network)
  @unimplemented
end

"""
    num_parameters(::Network)

Return the total number of parameters of a network.
"""
function num_parameters(::Network)
  @unimplemented
end

"""
    network_report(::Network) :: Report.Network

Return debugging informations on the network.
"""
function network_report(::Network) :: Report.Network
  @unimplemented
end

#####
##### Derived functions
#####

function evaluate(nn::Network, board, actions_mask)
  p, v = forward(nn, board)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p  = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1 .- sp
  return (p, v, p_invalid)
end

function MCTS.evaluate(nn::Network{G}, board, available_actions) where G
  x = GameInterface.vectorize_board(G, board)
  a = GameInterface.actions_mask(G, available_actions)
  xnet, anet = convert_input_tuple(nn, (x, a))
  p, v, _ = convert_output_tuple(nn, evaluate(nn, xnet, anet))
  return (p[a], v[1])
end

function MCTS.evaluate_batch(nn::Network{G}, batch) where G
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

end
