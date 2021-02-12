module Experiments

export Experiment

using ..AlphaZero

using Base: @kwdef

"""
    Experiment

A structure that contains the information necessary to replicate a training session.

# Constructor

    Experiment(gspec, params, mknet, netparams, benchmarks)

- `gspec` is the specification of the game to be played
- `params` has type [`Params`](@ref)
- `mknet` is a neural network constructor taking arguments `(netparams, gspec)`
- `netparams` are the neural network hyperparameters
- `benchmark` is a vector of [`Benchmark.Duel`](@ref) to be used as a benchmark to track
  training progress.                                                                                       
    
"""
@kwdef struct Experiment
  name :: String
  gspec :: AbstractGameSpec
  params :: Params
  mknet :: Any
  netparams :: Any
  benchmark :: Vector{<:Benchmark.Evaluation}
end

end