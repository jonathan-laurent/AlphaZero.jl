#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019-2021)
#####

module AlphaZero

  import Distributed
  using Formatting
  using Base: @kwdef
  using DataStructures: CircularBuffer
  using Distributions: Categorical, Dirichlet
  using Statistics: mean
  using Requires


  # Even when using the Knet backend, we use utilities from Flux such as
  # `Flux.batch` and `Flux.DataLoader`
  import Flux

  # When running on a CPU, having multiple threads does not play
  # well with BLAS multithreading
  import LinearAlgebra
  LinearAlgebra.BLAS.set_num_threads(1)

  # Internal helper functions
  include("util.jl")
  using .Util
  export Util
  export apply_temperature
  include("prof_utils.jl")
  using .ProfUtils

  # A generic interface for single-player or zero-sum two-players games.
  include("game.jl")
  using .GameInterface
  const GI = GameInterface
  export GameInterface, GI
  export AbstractGameEnv
  export AbstractGameSpec

  # A standalone, generic MCTS implementation
  include("mcts.jl")
  using .MCTS
  export MCTS

  # A generic network interface
  include("networks/network.jl")
  using .Network
  export Network
  export AbstractNetwork
  export OptimiserSpec
  export CyclicNesterov, Adam

  # Utilities to batch oracle calls
  include("batchifier.jl")
  using .Batchifier
  export Batchifier

  # Schedules
  include("schedule.jl")
  export AbstractSchedule
  export ConstSchedule, PLSchedule, StepSchedule, CyclicSchedule

  # Training params
  include("params.jl")
  export Params
  export MctsParams
  export SimParams
  export SelfPlayParams
  export LearningParams
  export ArenaParams
  export MemAnalysisParams
  export SamplesWeighingPolicy, CONSTANT_WEIGHT, LOG_WEIGHT, LINEAR_WEIGHT

  # Stats about training
  include("report.jl")
  export Report

  # Game traces
  include("trace.jl")
  export Trace
  export total_reward

  # Players and games
  include("play.jl")
  export AbstractPlayer, think, select_move, reset_player!, player_temperature
  export MctsPlayer
  export RandomPlayer
  export NetworkPlayer
  export PlayerWithTemperature
  export EpsilonGreedyPlayer
  export TwoPlayers
  export Human, interactive!
  export play_game

  # Utilities for distributed games simulation
  include("simulations.jl")
  export Simulator, simulate, simulate_distributed
  export record_trace
  export rewards_and_redundancy

  # Memory buffer to hold samples generated during self-play
  include("memory.jl")
  export MemoryBuffer, get_experience

  # Utilities to train the neural network based on collected samples
  include("learning.jl")

  # Main training algorithm
  include("training.jl")
  export Env, train!, initial_report
  export Handlers
  export AlphaZeroPlayer

  # A minmax player to be used as a baseline
  include("minmax.jl")
  using .MinMax
  export MinMax

  # Utilities to write benchmarks
  include("benchmark.jl")
  using .Benchmark
  export Benchmark

  # We provide a library of standard network, both in Knet and Flux.
  # Which backend is used to implement this library is determined during precompilation
  # based on the value of the ALPHAZERO_DEFAULT_DL_FRAMEWORK environment variable.
  const DEFAULT_DL_FRAMEWORK = get(ENV, "ALPHAZERO_DEFAULT_DL_FRAMEWORK", "FLUX")

  if DEFAULT_DL_FRAMEWORK == "FLUX"
    @info "Using the Flux implementation of AlphaZero.NetLib."
    @eval begin
      include("networks/flux.jl")
      const NetLib = FluxLib
    end
  elseif DEFAULT_DL_FRAMEWORK == "KNET"
    @info "Using the Knet implementation of AlphaZero.NetLib."
    @eval begin
      include("networks/knet.jl")
      const NetLib = KnetLib
    end
  else
    error("Unknown DL framework: $(DEFAULT_DL_FRAMEWORK)")
  end

  using .NetLib
  export NetLib
  export SimpleNet, SimpleNetHP, ResNet, ResNetHP

  # A structure that contains the information necessary to replicate a training session
  include("experiments.jl")
  using .Experiments
  export Experiments
  export Experiment

  # The default user interface is included here for convenience but it could be
  # replaced or separated from the main AlphaZero.jl package (which would also
  # enable dropping some dependencies such as Crayons or JSON3).
  include("ui/ui.jl")
  using .UserInterface
  const UI = UserInterface
  export UserInterface, UI
  export Session, resume!, save
  export explore

  # Bridge with CommonRLInterface.jl
  include("common_rl_intf.jl")
  export CommonRLInterfaceWrapper

  # A small library of standard examples
  include("examples.jl")
  export Examples

  # Scripts
  include("scripts/scripts.jl")
  export Scripts

  function __init__()
    # OpenSpiel.jl Wrapper
    @require OpenSpiel="ceb70bd2-fe3f-44f0-b81f-41608acaf2f2" begin
      include("openspiel.jl")
      export OpenSpielWrapper
      include("openspiel_example.jl")
      @info "AlphaZero.jl's OpenSpielWrapper loaded."
    end
  end


end