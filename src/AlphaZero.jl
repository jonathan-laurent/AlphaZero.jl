module AlphaZero

include("Util.jl")
include("GameInterface.jl")
include("MCTS.jl")

import .Util
import .GameInterface
import .MCTS

const GI = GameInterface

using Printf
using ProgressMeter
using DataStructures: Stack

include("Params.jl")
include("MemoryBuffer.jl")
include("Learning.jl")
include("Play.jl")
include("Coatch.jl")
include("Explorer.jl")

end
