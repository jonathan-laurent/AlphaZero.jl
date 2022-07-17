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

function uniform_mcts_tic_tac_toe(device, num_simulations=64)
    return MCTS.Policy(;
        oracle=MCTS.uniform_oracle, device=device, num_considered_actions=9, num_simulations
    )
end

function tic_tac_toe_winning_envs(; n_envs=2)
    return [bitwise_tictactoe_winning() for _ in 1:n_envs]
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
            qvalue_list = MCTS.completed_qvalues(tree, node, i)
            best = argmax(qvalue_list)

            best_move = 3
            oracle_prior = [0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0]
            oracle_value = Float32(0)

            @test isapprox(node.prior, oracle_prior, atol=1.0e-7)
            @test node.oracle_value == oracle_value
            @test length(qvalue_list) == length(node.children) == num_actions(env)
            @test best == best_move
        end
        function test_exploration(tree, envs::AbstractArray)
            (n_envs,) = size(envs)
            for i in 1:n_envs
                test_exploration(tree, envs[i], tree[i, 1], i)
            end
        end

        device = CPU()
        n_envs = 2

        policy = uniform_mcts_tic_tac_toe(device)
        envs = tic_tac_toe_winning_envs(; n_envs=n_envs)

        @testset "explore" begin
            tree = MCTS.explore(policy, envs)
            test_exploration(tree, envs)
        end
        @testset "gumbel explore" begin
            tree = MCTS.gumbel_explore(policy, envs, MersenneTwister(0))
            test_exploration(tree, envs)
            # All (valid) actions have been visited at least once
            # @test all(all(root.children[root.valid_actions] .!= 0) for root in tree[:, 1])
            # Only valid actions are explored
            @test all(
                !any(@. root.valid_actions == false && root.children > 0) for
                root in tree[:, 1]
            )
        end
    end
end

end
