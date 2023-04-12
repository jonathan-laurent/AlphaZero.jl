module Common

using Reexport

include("BitwiseTicTacToe.jl")
@reexport using .BitwiseTicTacToe

include("BitwiseConnectFour.jl")
@reexport using .BitwiseConnectFour

include("TestEnvs.jl")
@reexport using .TestEnvs

end
