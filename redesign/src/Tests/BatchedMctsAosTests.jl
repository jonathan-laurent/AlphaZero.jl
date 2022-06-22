module BatchedMctsAosTests

using ...BatchedMctsAos
using ...Util.Devices
using ..Common.BitwiseTicTacToe

using CUDA
using Test

export run_batched_mcts_tests

const MCTS = BatchedMctsAos

function run_batched_mcts_tests_on(device; num_simulations=2, num_envs=2)
    env = BitwiseTicTacToeEnv()
    mcts = MCTS.Policy(; device=device, oracle=nothing, num_simulations)
    tree = MCTS.explore(mcts, [env for _ in 1:num_envs])
    @test true
    return tree
end

function run_batched_mcts_tests()
    @testset "batched mcts compilation" begin
        run_batched_mcts_tests_on(CPU())
        CUDA.functional() && run_batched_mcts_tests_on(GPU())
    end
    return nothing
end

end
