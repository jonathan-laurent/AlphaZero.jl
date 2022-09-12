module BatchedMctsUtilityTests

using ...BatchedMctsUtility
using ..Common.Common

using Test

export run_batched_mcts_utility_tests

function run_batched_mcts_utility_tests()
    @testset "UniformTicTacToeEnvOracle" begin
        envs = [bitwise_tictactoe_winning()]
        check_oracle(UniformTicTacToeEnvOracle(), envs)
        @test true
    end
end

end
