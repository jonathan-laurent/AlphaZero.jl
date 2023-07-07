module BatchedMctsTests

using CUDA
using Flux: gpu
using Random: MersenneTwister
using Test

using ..Common.Common
using ...BatchedEnvs
using ...BatchedMcts
using ...BatchedMctsUtilities
using ...EnvOracles
using ...Network
using ...Util.Devices

import ..SimpleMctsTests as SimTest
import ...SimpleMcts as Sim


export run_batched_mcts_tests


const MCTS = BatchedMcts


function envs_array(constructor, device; n_envs)
    envs = [constructor() for _ in 1:n_envs]
    return DeviceArray(device)(envs)
end

function get_policy(device, oracle_fn, oracle_kwargs; num_simulations)
    oracle = oracle_fn(; oracle_kwargs...)
    return Policy(; device, oracle=oracle, num_simulations)
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
    mcts = get_policy(device, oracle_fn, oracle_kwargs; num_simulations=num_simulations)
    tree = MCTS.explore(mcts, envs)
    MCTS.gumbel_explore(mcts, envs, MersenneTwister(0))
    @test true
    return tree
end


function run_batched_mcts_tests()
    @testset "Compilation" test_compilation()
    @testset "Uniform exploration" test_uniform_exploration()
    @testset "Equivalence with Simple MCTS" test_equivalence_with_uniform_simple_mcts()
end


function test_compilation()
    rw1d_contructor = bitwise_random_walk_winning
    ttt_constructor = bitwise_tictactoe_winning

    uniform_kwargs = Dict()

    hp = SimpleNetHP(width=128, depth_common=2)
    rw1d_nn_kwargs_cpu = Dict(:nn => SimpleNet(10, 2, hp))
    rw1d_nn_kwargs_gpu = Dict(:nn => SimpleNet(10, 2, hp) |> gpu)
    ttt_nn_kwargs_cpu = Dict(:nn => SimpleNet(27, 9, hp))
    ttt_nn_kwargs_gpu = Dict(:nn => SimpleNet(27, 9, hp) |> gpu)

    @testset "CPU" begin
        @testset "Single-Player Env: RandomWalk1D" begin
            run_batched_mcts_tests_on(
                CPU(), rw1d_contructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                CPU(), rw1d_contructor, neural_network_env_oracle, rw1d_nn_kwargs_cpu
            )
        end
        @testset "Two-Player Env: TicTacToe" begin
            run_batched_mcts_tests_on(
                CPU(), ttt_constructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                CPU(), ttt_constructor, neural_network_env_oracle, ttt_nn_kwargs_cpu
            )
        end
    end

    CUDA.functional() && @testset "GPU" begin
        @testset "Single-Player Env: RandomWalk1D" begin
            run_batched_mcts_tests_on(
                GPU(), rw1d_contructor, uniform_env_oracle, uniform_kwargs
            )
            run_batched_mcts_tests_on(
                GPU(), rw1d_contructor, neural_network_env_oracle, rw1d_nn_kwargs_gpu
            )
        end
        @testset "Two-Player Env: TicTacToe" begin
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

    function test_exploration_correctness(tree, env, env_id)
        root = 1
        (; A, N, B) = MCTS.size(tree)
        tree_size = (Val(A), Val(N), Val(B))

        q_values = MCTS.completed_qvalues(tree, root, env_id, tree_size)
        @test length(q_values) == length(tree.children[:, root, env_id]) == num_actions(env)

        device = get_device(tree.policy_prior)
        policy_prior = DeviceArray(device)([0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0.2, 0])
        @test tree.policy_prior[:, root, env_id] ≈ policy_prior

        value_prior = Float32(0)
        @test tree.value_prior[root, env_id] == value_prior

        best = argmax(q_values)
        best_move = 3
        @test best == best_move
    end

    function test_exploration_correctness_all_envs(tree, envs)
        map(enumerate(envs)) do (env_id, env)
            test_exploration_correctness(tree, env, env_id)
        end
    end

    n_envs = 2
    n = 64

    @testset "Two-Player Env: TicTacToe" begin
        @testset "CPU" begin
            device = CPU()
            envs = envs_array(bitwise_tictactoe_winning, device; n_envs=n_envs)
            mcts = get_policy(device, uniform_env_oracle, Dict(); num_simulations=n)
            tree = MCTS.explore(mcts, envs)
            @testset "Correctness" test_exploration_correctness_all_envs(tree, envs)
        end
        CUDA.functional() && @testset "GPU" begin
            device = GPU()
            envs = envs_array(bitwise_tictactoe_winning, device; n_envs=n_envs)
            mcts = get_policy(device, uniform_env_oracle, Dict(); num_simulations=n)
            tree = MCTS.explore(mcts, envs)
            @testset "Correctness" test_exploration_correctness_all_envs(tree, envs)
        end
    end
end


function test_equivalence_with_uniform_simple_mcts()
    device = GPU()

    simple_mcts_policy = SimTest.uniform_mcts_policy(; num_simulations=64)
    simple_mcts_env = tictactoe_winning()
    simple_mcts_tree = Sim.explore(simple_mcts_policy, simple_mcts_env)
    sim_qvalues = Sim.completed_qvalues(simple_mcts_tree)

    batch_mcts_policy = get_policy(device, uniform_env_oracle, Dict(); num_simulations=64)
    batch_mcts_envs = envs_array(bitwise_tictactoe_winning, device; n_envs=2)
    batch_mcts_tree = MCTS.explore(batch_mcts_policy, batch_mcts_envs)

    (; A, N, B) = MCTS.size(batch_mcts_tree)
    tree_size = (Val(A), Val(N), Val(B))

    batch_mcts_qvalues_inf = MCTS.completed_qvalues(batch_mcts_tree, 1, 1, tree_size)
    batch_mcts_qvalues = [q for q in batch_mcts_qvalues_inf if q != -Inf32]

    @test sim_qvalues ≈ batch_mcts_qvalues

end

end
