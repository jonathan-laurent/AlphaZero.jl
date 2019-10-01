module AlphaZero

include("Util.jl")
include("GameInterface.jl")
include("SimpleMCTS.jl")
include("MCTS.jl")

import .Util
import .GameInterface
import .MCTS

const GI = GameInterface

using Printf
using ProgressMeter

include("Params.jl")
include("MemoryBuffer.jl")
include("Learning.jl")
include("Play.jl")
include("Coatch.jl")

end
