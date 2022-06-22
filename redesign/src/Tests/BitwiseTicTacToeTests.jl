module BitwiseTicTacToeTests

using Test
using ..BatchedEnvsTests
using ..Common.BitwiseTicTacToe
using ReinforcementLearningEnvironments

export run_bitwise_tictactoe_tests

function run_bitwise_tictactoe_tests()
    @testset "bitwise tictactoe" begin
        test_equivalent(BitwiseTicTacToeEnv, TicTacToeEnv)
        test_batch_simulate(BitwiseTicTacToeEnv)
    end
    return nothing
end

end
