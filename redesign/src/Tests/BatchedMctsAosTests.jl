module BatchedMctsAosTests

using ...BatchedMctsAos
using ...SimpleMcts: RolloutOracle, uniform_oracle
using ...Util.Devices
using ..Common.BitwiseTicTacToe

using Random: MersenneTwister
using CUDA
using Test
using JET
using ReinforcementLearningEnvironments

export run_batched_mcts_tests

const MCTS = BatchedMctsAos

function run_batched_mcts_tests_on(device; num_simulations=2, num_envs=2)
    env = BitwiseTicTacToeEnv()
    mcts = MCTS.Policy(; device=device, oracle=nothing, num_simulations)
    tree = MCTS.explore(mcts, [env for _ in 1:num_envs])
    @test true
    return tree
end

function mcts_rollout(device)
    return Policy{RolloutOracle(MersenneTwister(0)),device}
end

function random_walk_envs(; n_envs=2)
    return [RandomWalk1D() for _ in 1:n_envs]
end

function profile_explore()
    policy = mcts_rollout(CPU())
    envs = random_walk_envs()
    return explore(policy, envs)
end

function run_batched_mcts_tests()
    @testset "batched mcts compilation" begin
        run_batched_mcts_tests_on(CPU())
        CUDA.functional() && run_batched_mcts_tests_on(GPU())
    end
    @testset "batched mcts inferred" begin
        @test_opt profile_explore()
    end
end

end
