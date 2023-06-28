module BitwiseTicTacToeTests

using Test
using ..BatchedEnvsTests
using ..Common.BitwiseTicTacToe
using ReinforcementLearningEnvironments: TicTacToeEnv

export run_bitwise_tictactoe_tests

function run_bitwise_tictactoe_tests()
    Env = BitwiseTicTacToeEnv
    @testset "bitwise tictactoe" begin
        @testset "RLEnvs equivalence" test_equivalent(Env, TicTacToeEnv)
        @testset "bitwise batch simulation" test_batch_simulate(Env)
        @testset "GPU friendliness" test_gpu_friendliness(Env; num_actions=5)
    end
    return nothing
end

end
