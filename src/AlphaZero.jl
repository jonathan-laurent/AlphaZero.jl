#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019-2020)
#####

module AlphaZero

# Submodules
export MCTS, MinMax, GameInterface, GI, Report, Network, Benchmark
# AlphaZero parameters
export Params, SelfPlayParams, LearningParams, ArenaParams
export MctsParams, MemAnalysisParams
export SamplesWeighingPolicy, CONSTANT_WEIGHT, LOG_WEIGHT, LINEAR_WEIGHT
export AbstractSchedule, ConstSchedule, PLSchedule, StepSchedule
# Players and games
export AbstractGame, AbstractPlayer, TwoPlayers, Trace
export think, select_move, reset_player!, player_temperature, apply_temperature
export play_game, interactive!
export MctsPlayer, RandomPlayer, EpsilonGreedyPlayer, NetworkPlayer, Human
export ColorPolicy, ALTERNATE_COLORS, BASELINE_WHITE, CONTENDER_WHITE
# Networks
export AbstractNetwork, OptimiserSpec, Nesterov, CyclicNesterov, Adam
export SimpleNet, SimpleNetHP, ResNet, ResNetHP
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

include("mcts_simple.jl")
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

# We provide a library of predefined network architectures for convenience.
# Right now, it is included in the main AlphaZero.jl package. In the future,
# we may want to separate it so as to drop the Knet and Flux dependencies.

const USE_KNET_FOR_NETLIB = false # The Flux netlib is currently broken

if USE_KNET_FOR_NETLIB
  @eval begin
    include("networks/knet.jl")
    using .KNets
  end
else
  @eval begin
    include("networks/flux.jl")
    using .FluxNets
  end
end

# The default user interface is included here for convenience but it could be
# replaced or separated from the main AlphaZero.jl package (which would also
# enable dropping some dependencies such as Crayons or JSON3).

include("ui/ui.jl")
using .UserInterface

end
