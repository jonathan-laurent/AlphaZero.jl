#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019-2020)
#####

module AlphaZero

# Submodules
export MCTS, MinMax, GameInterface, GI, Report, Benchmark
export Network, KnetLib, FluxLib, NetLib
# AlphaZero parameters
export Params, SelfPlayParams, LearningParams, ArenaParams
export MctsParams, MemAnalysisParams
export SamplesWeighingPolicy, CONSTANT_WEIGHT, LOG_WEIGHT, LINEAR_WEIGHT
export AbstractSchedule, ConstSchedule, PLSchedule, StepSchedule
# Players and games
export AbstractGame, AbstractPlayer, TwoPlayers, Trace
export think, select_move, reset_player!, player_temperature, apply_temperature
export play_game, interactive!, total_reward
export MctsPlayer, RandomPlayer, EpsilonGreedyPlayer, NetworkPlayer, Human
export ColorPolicy, ALTERNATE_COLORS, BASELINE_WHITE, CONTENDER_WHITE
# Networks
export AbstractNetwork, OptimiserSpec, Nesterov, CyclicNesterov, Adam
# Training environments
export Env, train!, get_experience
# User interface
export UserInterface
export Session, resume!, save, play_interactive_game
export Explorer, start_explorer

include("util.jl")
import .Util
using .Util: Option, apply_temperature

include("game.jl")
using .GameInterface
const GI = GameInterface

include("mcts.jl")
import .MCTS

include("networks/network.jl")
using .Network

include("batchifier.jl")
import .Batchifier

using Formatting
using Base: @kwdef
using DataStructures: CircularBuffer
using Distributions: Categorical, Dirichlet
using Statistics: mean

import Distributed

include("schedule.jl")
include("params.jl")
include("report.jl")
include("trace.jl")
include("memory.jl")
include("learning.jl")
include("play.jl")
include("training.jl")

include("minmax.jl")
import .MinMax

include("benchmark.jl")
using .Benchmark

include("networks/knet.jl")
# import .KnetLib

include("networks/flux.jl")
# import .FluxLib

# The default user interface is included here for convenience but it could be
# replaced or separated from the main AlphaZero.jl package (which would also
# enable dropping some dependencies such as Crayons or JSON3).

include("ui/ui.jl")
using .UserInterface

# Choose the default DL framework based on an environment variable
function __init__()
  @eval begin
    const DEFAULT_DL_FRAMEWORK = get(ENV, "ALPHAZERO_DEFAULT_DL_FRAMEWORK", "FLUX")
    const NetLib =
      if DEFAULT_DL_FRAMEWORK == "FLUX"
        @info "Using Flux."
        FluxLib
      elseif DEFAULT_DL_FRAMEWORK == "KNET"
        @info "Using Knet."
        KnetLib
      else
        error("Unknown DL framework: $(DEFAULT_DL_FRAMEWORK)")
      end
  end
end

end
