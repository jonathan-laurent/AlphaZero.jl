module BatchedMctsTests

using CUDA
using Random: MersenneTwister
using Test

using ...BatchedMctsUtilities
using ...EnvOracles
using ...BatchedMcts
using ...Util.Devices
using ...BatchedEnvs
using ..Common.Common

import ...SimpleMcts as Sim
import ..SimpleMctsTests as SimTest


export run_batched_mcts_tests


const MCTS = BatchedMcts


function tic_tac_toe_winning_envs(; n_envs=2)
    return [bitwise_tictactoe_winning() for _ in 1:n_envs]
end

# function random_walk_winning_envs(; n_envs=2)
#     return [bitwise_random_walk_winning() for _ in 1:n_envs]
# end

function uniform_mcts_tictactoe(device; num_simulations=64)
    oracle = uniform_env_oracle()
    # oracle = neural_network_env_oracle()
    return Policy(; device, oracle=oracle, num_simulations)
end

# function uniform_mcts_random_walk(device; num_simulations=64)
#     oracle = UniformRandomWalk1DEnvOracle()
#     return Policy(; device, oracle=oracle, num_simulations)
# end

function run_batched_mcts_tests_on(device::Device)
    envs = tic_tac_toe_winning_envs()
    mcts = uniform_mcts_tictactoe(device)
    tree = MCTS.explore(mcts, envs)
    tree = MCTS.gumbel_explore(mcts, envs, MersenneTwister(0))
    @test true
    return tree
end

# function run_batched_mcts_tests_random_walk_on(device::Device)
#     envs = random_walk_winning_envs()
#     mcts = uniform_mcts_random_walk(device)
#     tree = MCTS.explore(mcts, envs)
#     tree = MCTS.gumbel_explore(mcts, envs, MersenneTwister(0))
#     @test true
#     return tree
# end

function run_batched_mcts_tests()
    @testset "Batched Mcts" begin
        @testset "Compilation" begin
            run_batched_mcts_tests_on(CPU())
            # run_batched_mcts_tests_random_walk_on(CPU())
            CUDA.functional() && run_batched_mcts_tests_on(GPU())
            # CUDA.functional() && run_batched_mcts_tests_random_walk_on(GPU())
        end
        @testset "Policy" begin
            function test_exploration(tree, env, bid)
                root = 1
                (; A, N, B) = MCTS.size(tree)
                tree_size = (Val(A), Val(N), Val(B))

                qvalue_list = MCTS.completed_qvalues(tree, root, bid, tree_size)
                best = argmax(qvalue_list)

                best_move = 3
                policy_prior = Devices.CuArray([0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0])
                value_prior = Float32(0)

                @test isapprox(tree.policy_prior[:, root, bid], policy_prior, atol=1.0e-7)
                @test tree.value_prior[root, bid] == value_prior
                @test length(qvalue_list) ==
                    length(tree.children[:, root, bid]) ==
                    num_actions(env)
                @test best == best_move
            end
            function test_exploration(tree, envs::AbstractArray)
                n_envs = length(envs)
                for bid in 1:n_envs
                    test_exploration(tree, envs[bid], bid)
                end
            end

            device = GPU()
            n_envs = 2

            envs = tic_tac_toe_winning_envs()
            mcts = uniform_mcts_tictactoe(device)

            @testset "Explore" begin
                tree = MCTS.explore(mcts, envs)
                test_exploration(tree, envs)
            end
            @testset "Gumbel explore" begin
                tree = MCTS.gumbel_explore(mcts, envs, MersenneTwister(0))
                test_exploration(tree, envs)
                # All (valid) actions have been visited at least once
                @test all(@. tree.valid_actions[:, 1, :] || tree.children[:, 1, :] == 0)
                # Only valid actions are explored
                @test all(@. !tree.valid_actions[:, 1, :] || tree.children[:, 1, :] != 0)
            end
        end
        @testset "Equivalence with SimpleMcts" begin
            sim_policy = SimTest.uniform_mcts_policy(; n=64)
            sim_env = tictactoe_winning()
            sim_tree = Sim.explore(sim_policy, sim_env)

            bat_policy = uniform_mcts_tictactoe(GPU())
            bat_env = tic_tac_toe_winning_envs(; n_envs=2)
            bat_tree = MCTS.explore(bat_policy, bat_env)

            (; A, N, B) = MCTS.size(bat_tree)
            tree_size = (Val(A), Val(N), Val(B))

            sim_qvalues = Sim.completed_qvalues(sim_tree)
            bat_qvalues_inf = MCTS.completed_qvalues(bat_tree, 1, 1, tree_size)
            bat_qvalues = [q for q in bat_qvalues_inf if q != -Inf32]

            println("sim_qvalues: ", sim_qvalues)
            println("bat_qvalues: ", bat_qvalues)
            println("\n\n\n")

            @test isapprox(sim_qvalues, bat_qvalues, atol=1e-6)
        end
    end
end

end
