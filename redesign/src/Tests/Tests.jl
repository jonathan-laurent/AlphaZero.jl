module Tests

using Reexport
using Test

export run_all_tests

include("BatchedEnvsTests.jl")
@reexport using .BatchedEnvsTests

include("BitwiseTicTacToe.jl")
@reexport using .BitwiseTicTacToe

include("BitwiseTicTacToeTests.jl")
@reexport using .BitwiseTicTacToeTests

include("UtilTests.jl")
@reexport using .UtilTests

include("TestEnvs.jl")
@reexport using .TestEnvs

include("MctsTests.jl")
@reexport using .MctsTests

function run_all_tests()
    @testset "RLZero tests" begin
        run_util_tests()
        run_bitwise_tictactoe_tests()
        run_mcts_tests()
    end
    return nothing
end

end
