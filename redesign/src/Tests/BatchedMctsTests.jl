module BatchedMctsTests

using ...BatchedMcts
using ...Util.Devices
using ...BatchedEnvs
using ..Common.Common
import ...SimpleMcts as Sim
import ..SimpleMctsTests as SimTest

using CUDA
using Test

export run_batched_mcts_tests

const MCTS = BatchedMcts

function tic_tac_toe_winning_envs(; n_envs=2)
    return [bitwise_tictactoe_winning() for _ in 1:n_envs]
end

function uniform_mcts_tictactoe(device; num_simulations=64)
    return Policy(; device, oracle=UniformTicTacToeEnvOracle(), num_simulations)
end

function run_batched_mcts_tests_on(device::Device)
    envs = tic_tac_toe_winning_envs()
    mcts = uniform_mcts_tictactoe(device)
    tree = MCTS.explore(mcts, envs)
    @test true
    return tree
end

function run_batched_mcts_tests()
    @testset "Batched Mcts" begin
        @testset "Compilation" begin
            run_batched_mcts_tests_on(CPU())
            CUDA.functional() && run_batched_mcts_tests_on(GPU())
        end
        @testset "UniformTicTacToeEnvOracle" begin
            envs = tic_tac_toe_winning_envs()
            aids = [2 for env in envs]
            check_oracle(UniformTicTacToeEnvOracle(), envs, aids)
            @test true
        end
        @testset "Exploration" begin
            function test_exploration(tree, env, bid)
                root = 1
                qvalue_list = MCTS.completed_qvalues(tree, root, bid)
                best = argmax(qvalue_list)

                best_move = 3
                policy_prior = [0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0]
                value_prior = Float32(0)

                @test isapprox(tree.policy_prior[:, root, bid], policy_prior, atol=1.0e-7)
                @test tree.value_prior[root, bid] == value_prior
                @test length(qvalue_list) ==
                    length(tree.children[:, root, bid]) ==
                    num_actions(env)
                @test best == best_move
            end

            device = CPU()
            n_envs = 2

            envs = tic_tac_toe_winning_envs()
            mcts = uniform_mcts_tictactoe(device)
            tree = MCTS.explore(mcts, envs)

            for bid in 1:n_envs
                test_exploration(tree, envs[bid], bid)
            end
        end
        @testset "Equivalence with SimpleMcts" begin
            sim_policy = SimTest.uniform_mcts_policy_tic_tac_toe(; num_simulations=64)
            sim_env = tictactoe_winning()
            sim_tree = Sim.explore(sim_policy, sim_env)

            bat_policy = uniform_mcts_tictactoe(CPU())
            bat_env = tic_tac_toe_winning_envs(; n_envs=1)
            bat_tree = MCTS.explore(bat_policy, bat_env)

            sim_qvalues = Sim.completed_qvalues(sim_tree)
            bat_qvalues_inf = MCTS.completed_qvalues(bat_tree, 1, 1)
            bat_qvalues = [q for q in bat_qvalues_inf if q != -Inf32]

            @test isapprox(sim_qvalues, bat_qvalues, atol=1e-6)
        end
    end
end

end
