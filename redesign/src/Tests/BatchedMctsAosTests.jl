module BatchedMctsAosTests

using ...BatchedMctsAos
using ...Util.Devices
using ...BatchedEnvs
using ..Common.Common

using Random: MersenneTwister
using CUDA
using Test
using JET
using ReinforcementLearningEnvironments

export run_batched_mcts_tests

const MCTS = BatchedMctsAos

function run_batched_mcts_tests_on(device; num_simulations=2, num_envs=2)
    env = BitwiseTicTacToeEnv()
    mcts = MCTS.Policy(; device=device, oracle=uniform_oracle, num_simulations)
    tree = MCTS.explore(mcts, [env for _ in 1:num_envs])
    @test true
    return tree
end

function rollout_mcts_randomwalk1d(device)
    return Policy(;
        oracle=RolloutOracle(MersenneTwister(0)), device=device, num_considered_actions=2
    )
end

function uniform_mcts_tic_tac_toe(device, num_simulations=64)
    return Policy(;
        oracle=uniform_oracle, device=device, num_considered_actions=9, num_simulations
    )
end

function tic_tac_toe_winning_envs(; n_envs=2)
    return [bitwise_tictactoe_winning() for _ in 1:n_envs]
end

function random_walk_envs(; n_envs=2)
    return [RandomWalk1D() for _ in 1:n_envs]
end

function profile_explore()
    policy = rollout_mcts_randomwalk1d(CPU())
    envs = random_walk_envs()
    return explore(policy, envs)
end

function run_batched_mcts_tests()
    @testset "batched mcts compilation" begin
        run_batched_mcts_tests_on(CPU())
        CUDA.functional() && run_batched_mcts_tests_on(GPU())
    end
    @testset "batched mcts oracle" begin
        @testset "uniform_oracle" begin
            env = bitwise_tictactoe_winning()
            prior, value = uniform_oracle(env)

            @test value == 0
            @test prior == ones(Float32, num_actions(env)) ./ num_actions(env)
        end
    end
    @testset "batched mcts policy" begin
        function test_exploration(tree, env, node, i)
            qvalue_list = completed_qvalues(tree, node, i)
            best = argmax(qvalue_list)

            best_move = 3
            oracle_prior = [0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0]
            oracle_value = Float32(0)

            @test isapprox(node.prior, oracle_prior, atol=1.0e-7)
            @test node.oracle_value == oracle_value
            @test length(qvalue_list) == length(node.children) == num_actions(env)
            @test best == best_move
        end

        device = CPU()
        n_envs = 2

        policy = uniform_mcts_tic_tac_toe(device)
        envs = tic_tac_toe_winning_envs(; n_envs=n_envs)
        tree = explore(policy, envs)

        for i in 1:n_envs
            test_exploration(tree, envs[i], tree[i, 1], i)
        end
    end
end

end
