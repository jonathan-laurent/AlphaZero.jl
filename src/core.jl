module Training

  # AlphaZero parameters
  export Params, SelfPlayParams, LearningParams, ArenaParams
  export MctsParams, MemAnalysisParams
  export SamplesWeighingPolicy, CONSTANT_WEIGHT, LOG_WEIGHT, LINEAR_WEIGHT
  export AbstractSchedule, ConstSchedule, PLSchedule, StepSchedule
  
  # Training environments
  export Env, train!
  export MemoryBuffer, get_experience
  export Report

  using Formatting
  using Base: @kwdef
  using DataStructures: CircularBuffer
  using Distributions: Categorical, Dirichlet
  using Statistics: mean

  import Distributed

  using AlphaZero.GameInterface
  using AlphaZero.Network
  using AlphaZero.Batchifier
  using AlphaZero.Util: Option, apply_temperature
  using AlphaZero: Util, GI, Network, Batchifier, MCTS

  include("schedule.jl")
  include("params.jl")
  include("report.jl")
  include("memory.jl")
  include("learning.jl")
  include("training.jl")

end