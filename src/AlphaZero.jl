#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019)
#####

module AlphaZero

export Session

include("Util.jl")
import .Util
using .Util: Option

include("GameInterface.jl")
import .GameInterface
const GI = GameInterface

include("MCTS.jl")
import .MCTS

include("Log.jl")
using .Log

include("Report.jl")

include("Network.jl")
using .Networks

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

include("Params.jl")
include("MemoryBuffer.jl")
include("Learning.jl")
include("Play.jl")
include("Training.jl")
include("Explorer.jl")
include("Validation.jl")
include("Plots.jl")
include("Session.jl")

# We add default support for the Flux.jl framework
include("Flux.jl")

end

# External resources on AlphaZero and MCTS:
# + https://web.stanford.edu/~surag/posts/alphazero.html
# + https://int8.io/monte-carlo-tree-search-beginners-guide/
# + https://medium.com/oracledevs/lessons-from-alpha-zero
