module Common

using Reexport

include("Envs/BitwiseRandomWalk1D.jl")
@reexport using .BitwiseRandomWalk1D

include("Envs/BitwiseTicTacToe.jl")
@reexport using .BitwiseTicTacToe

include("Envs/BitwiseConnectFour.jl")
@reexport using .BitwiseConnectFour

include("TestEnvs.jl")
@reexport using .TestEnvs

include("Evaluation/Heuristics/BitwiseTicTacToeHeuristic.jl")
@reexport using .BitwiseTicTacToeHeuristic

include("Evaluation/EvaluationFunctions/BitwiseTicTacToeEvaluationFns.jl")
@reexport using .BiwtiseTicTacToeEvalFns

end
