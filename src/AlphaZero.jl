#####
##### AlphaZero.jl
##### Jonathan Laurent, Carnegie Mellon University (2019)
#####

module AlphaZero

include("Util.jl")
import .Util

include("GameInterface.jl")
import .GameInterface
const GI = GameInterface

include("MCTS.jl")
import .MCTS

include("Log.jl")
using .Log

using CUDAapi
if has_cuda()
  try
    using CuArrays
    @eval const CUARRAYS_IMPORTED = true
  catch ex
    @warn(
      "CUDA is installed, but CuArrays.jl fails to load.",
      exception=(ex,catch_backtrace()))
    @eval const CUARRAYS_IMPORTED = false
  end
else
  @eval const CUARRAYS_IMPORTED = false
end

import Flux
import Plots

using Printf
using Formatting
using Crayons
using ProgressMeter
using Base: @kwdef
using DataStructures: Stack, CircularBuffer
using Distributions: Categorical, Dirichlet
using Flux: Tracker, Chain, Dense, relu, softmax, cpu, gpu
using Statistics: mean

include("Params.jl")
include("Report.jl")
include("MemoryBuffer.jl")
include("Learning.jl")
include("Play.jl")
include("Training.jl")
include("Explorer.jl")

end

# External resources on AlphaZero and MCTS:
# + https://web.stanford.edu/~surag/posts/alphazero.html
# + https://int8.io/monte-carlo-tree-search-beginners-guide/
# + https://medium.com/oracledevs/lessons-from-alpha-zero
