#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019)
#####

module AlphaZero

export MCTS, GameInterface, GI, Report, Network, FluxNets
export RolloutsValidation
export AbstractSchedule, PLSchedule
export Params, SelfPlayParams, ArenaParams, MctsParams, LearningParams
export Env, train!, learning!, self_play!, memory_report, get_experience
export Session, resume!, save, plot_learning

include("Util.jl")
import .Util
using .Util: Option, @unimplemented

include("GameInterface.jl")
import .GameInterface
const GI = GameInterface

include("MCTS.jl")
import .MCTS

include("Log.jl")
using .Log

include("Network.jl")
using .Network

import Plots
import Colors
import JSON2

using Formatting
using Crayons
using Colors: @colorant_str
using ProgressMeter
using Base: @kwdef
using Serialization: serialize, deserialize
using DataStructures: Stack, CircularBuffer
using Distributions: Categorical, Dirichlet
using Statistics: mean

include("Schedule.jl")
include("Params.jl")
include("Report.jl")
include("MemoryBuffer.jl")
include("Learning.jl")
include("Play.jl")
include("Training.jl")
include("Explorer.jl")
include("Validation.jl")
include("Plots.jl")
include("Session.jl")

# We add default support for the Flux.jl framework
include("Flux/FluxNets.jl")
using .FluxNets

end

# External resources on AlphaZero and MCTS:
# + https://web.stanford.edu/~surag/posts/alphazero.html
# + https://int8.io/monte-carlo-tree-search-beginners-guide/
# + https://medium.com/oracledevs/lessons-from-alpha-zero
