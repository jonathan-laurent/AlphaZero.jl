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

include("Evaluation/EvaluationFunctions/BitwiseRandomWalk1DEvaluationFns.jl")
@reexport using .BitwiseRandomWalk1DEvalFns

include("Evaluation/Heuristics/BitwiseTicTacToeHeuristic.jl")
@reexport using .BitwiseTicTacToeHeuristic

include("Evaluation/Heuristics/BitwiseConnectFourHeuristic.jl")
@reexport using .BitwiseConnectFourHeuristic

include("Evaluation/EvaluationFunctions/BitwiseTicTacToeEvaluationFns.jl")
@reexport using .BitwiseTicTacToeEvalFns

include("Evaluation/EvaluationFunctions/BitwiseConnectFourEvaluationFns.jl")
@reexport using .BitwiseConnectFourEvalFns

end
