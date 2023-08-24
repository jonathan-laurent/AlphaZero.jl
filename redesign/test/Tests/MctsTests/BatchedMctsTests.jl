module BatchedMctsTests

using CUDA
using Flux: gpu
using Random: MersenneTwister
using Test

using ...Common.Common
using ...Common.BitwiseRandomWalk1D
using ...Common.BitwiseTicTacToe
using ....BatchedEnvs
using ....BatchedMcts
using ....BatchedMctsUtilities
using ....EnvOracles
using ....Network
using ....Util.Devices

import ..SimpleMctsTests as SimpleMctsTests
import ....SimpleMcts


export run_batched_mcts_tests


function run_batched_mcts_tests()
    @testset "Compilation" test_compilation()
    @testset "Uniform exploration" test_uniform_exploration()
    @testset "Equivalence with Simple MCTS" test_equivalence_with_uniform_simple_mcts()
end

function envs_array(constructor, device; n_envs)
    envs = [constructor() for _ in 1:n_envs]
    return DeviceArray(device)(envs)
end

function get_mcts_config(device, oracle_fn, oracle_kwargs; num_simulations)
    oracle = oracle_fn(; oracle_kwargs...)
    return GumbelMctsConfig(; device, oracle, num_simulations)
end

function run_batched_mcts_tests_on(
    device::Device,
    constructor,
    oracle_fn,
    oracle_kwargs;
    n_envs=3,
    num_simulations=64
)
    envs = envs_array(constructor, device; n_envs=n_envs)
    mcts = get_mcts_config(device, oracle_fn, oracle_kwargs; num_simulations)
    tree = BatchedMcts.explore(mcts, envs)
    BatchedMcts.gumbel_explore(mcts, envs, MersenneTwister(0))
    @test true
    return tree
end

function test_compilation()
    rw1d_contructor = TestEnvs.bitwise_random_walk_winning
    ttt_constructor = TestEnvs.bitwise_tictactoe_winning

    rw1d_state_size = BatchedEnvs.state_size(BitwiseRandomWalk1DEnv)
    rw1d_na = BatchedEnvs.num_actions(BitwiseRandomWalk1DEnv)
    ttt_state_size = BatchedEnvs.state_size(BitwiseTicTacToeEnv)
    ttt_na = BatchedEnvs.num_actions(BitwiseTicTacToeEnv)

    uniform_kwargs = Dict()

    hp = SimpleNetHP(width=128, depth_common=2)

    @testset "CPU" begin
        rw1d_nn_kwargs_cpu = Dict(:nn => SimpleNet(rw1d_state_size..., rw1d_na, hp))
        ttt_nn_kwargs_cpu = Dict(:nn => SimpleNet(ttt_state_size..., ttt_na, hp))

        @testset "Single-Player Env: BitwiseRandomWalk1D" begin
            run_batched_mcts_tests_on(
                CPU(), rw1d_contructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                CPU(), rw1d_contructor, neural_network_env_oracle, rw1d_nn_kwargs_cpu
            )
        end
        @testset "Two-Player Env: BitwiseTicTacToe" begin
            run_batched_mcts_tests_on(
                CPU(), ttt_constructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                CPU(), ttt_constructor, neural_network_env_oracle, ttt_nn_kwargs_cpu
            )
        end
    end

    CUDA.functional() && @testset "GPU" begin
        rw1d_nn_kwargs_gpu = Dict(:nn => SimpleNet(rw1d_state_size..., rw1d_na, hp) |> gpu)
        ttt_nn_kwargs_gpu = Dict(:nn => SimpleNet(ttt_state_size..., ttt_na, hp) |> gpu)

        @testset "Single-Player Env: BitwiseRandomWalk1D" begin
            run_batched_mcts_tests_on(
                GPU(), rw1d_contructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                GPU(), rw1d_contructor, neural_network_env_oracle, rw1d_nn_kwargs_gpu
            )
        end
        @testset "Two-Player Env: BitwiseTicTacToe" begin
            run_batched_mcts_tests_on(
                GPU(), ttt_constructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                GPU(), ttt_constructor, neural_network_env_oracle, ttt_nn_kwargs_gpu
            )
        end
    end
end

function test_uniform_exploration()

    function test_exploration_correctness(tree, mcts, env_id, envs)
        q_values = BatchedMcts.get_completed_qvalues(tree, mcts)[:, env_id]
        @test length(q_values) == length(tree.children[:, BatchedMcts.ROOT, env_id])
        @test length(q_values) == BatchedEnvs.num_actions(eltype(envs))

        device = get_device(tree.policy_prior)
        policy_prior = DeviceArray(device)([0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0])
        @test tree.policy_prior[:, BatchedMcts.ROOT, env_id] ≈ policy_prior

        value_prior = Float32(0)
        tree_vp_cpu = Array(tree.value_prior)  # convert to CPU to avoid scalar indexing
        @test tree_vp_cpu[BatchedMcts.ROOT, env_id] == value_prior

        best = argmax(Array(q_values))  # convert to CPU to avoid scalar indexing
        best_move = 3
        @test best == best_move
    end

    function test_exploration_correctness_all_envs(tree, envs, mcts)
        num_envs = length(envs)
        map(1:num_envs) do env_id
            test_exploration_correctness(tree, mcts, env_id, envs)
        end
    end

    n_envs = 2
    n = 64

    @testset "Two-Player Env: BitwiseTicTacToe" begin
        @testset "CPU" begin
            device = CPU()
            envs = envs_array(bitwise_tictactoe_winning, device; n_envs=n_envs)
            mcts = get_mcts_config(device, uniform_env_oracle, Dict(); num_simulations=n)
            tree = BatchedMcts.explore(mcts, envs)
            @testset "Correctness" test_exploration_correctness_all_envs(tree, envs, mcts)
        end
        CUDA.functional() && @testset "GPU" begin
            device = GPU()
            envs = envs_array(bitwise_tictactoe_winning, device; n_envs=n_envs)
            mcts = get_mcts_config(device, uniform_env_oracle, Dict(); num_simulations=n)
            tree = BatchedMcts.explore(mcts, envs)
            @testset "Correctness" test_exploration_correctness_all_envs(tree, envs, mcts)
        end
    end
end

function test_equivalence_with_uniform_simple_mcts()
    device = GPU()

    simple_mcts_config = SimpleMctsTests.uniform_mcts_policy(; num_simulations=64)
    simple_mcts_env = tictactoe_winning()
    simple_mcts_tree = SimpleMcts.explore(simple_mcts_config, simple_mcts_env)
    sim_qvalues = SimpleMcts.completed_qvalues(simple_mcts_tree)

    mcts_config = get_mcts_config(device, uniform_env_oracle, Dict(); num_simulations=64)
    batch_mcts_envs = envs_array(bitwise_tictactoe_winning, device; n_envs=2)
    batch_mcts_tree = BatchedMcts.explore(mcts_config, batch_mcts_envs)

    batch_mcts_qvalues_inf = BatchedMcts.get_completed_qvalues(batch_mcts_tree, mcts_config)
    batch_mcts_qvalues = [q for q in Array(batch_mcts_qvalues_inf)[:, 1] if q != -Inf32]

    @test sim_qvalues ≈ batch_mcts_qvalues
end

end
