module BatchedMctsTests

using ...BatchedMcts
using ...Util.Devices
using ...BatchedEnvs
using ..Common.Common

using CUDA
using Test
using ReinforcementLearningEnvironments
using ReinforcementLearningBase

export run_batched_mcts_tests

const MCTS = BatchedMcts

function tic_tac_toe_winning_envs(; n_envs=2)
    return [tictactoe_winning() for _ in 1:n_envs]
end

function uniform_mcts_tictactoe(device, num_simulations=64)
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
            aids = [legal_action_space(env)[1] for env in envs]
            check_oracle(UniformTicTacToeEnvOracle(), envs, aids)
            @test true
        end
        @testset "Exploration" begin end
    end
end

end
