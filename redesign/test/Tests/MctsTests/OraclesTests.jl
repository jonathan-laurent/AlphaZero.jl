module OraclesTests

using CUDA
using Flux: gpu
using Random: MersenneTwister
using Test

using ...Common.Common
using ...Common.BitwiseRandomWalk1D
using ...Common.BitwiseTicTacToe
using ....BatchedEnvs
using ....BatchedMctsUtilities
using ....EnvOracles
using ....Network
using ....Util.Devices

import ..SimpleMctsTests as SimpleMctsTests
import ....SimpleMcts


export run_oracle_tests


function run_oracle_tests()
    @testset "Uniform EnvOracle" test_uniform_env_oracle()
    @testset "Neural Network EnvOracle" test_neural_network_env_oracle()
end

function _check_oracle(oracle::EnvOracle, envs)
    check_keys(keys, ref_keys) = Set(keys) == Set(ref_keys)
    device = get_device(envs)
    B = length(envs)

    # init_fn()
    init_fn_out = oracle.init_fn(envs)

    @testset "init_fn(): Named-tuple check" begin
        exp = (:internal_states, :valid_actions, :logit_prior, :policy_prior, :value_prior)
        msg = "The `EnvOracle`'s `init_fn()` function should returned a named-tuple with " *
              "the following fields: internal_states, valid_actions, logit_prior, " *
              "policy_prior, value_prior."
        @assert check_keys(keys(init_fn_out), exp) msg
        @test true
    end

    valid_actions = init_fn_out.valid_actions
    A, _ = size(init_fn_out.valid_actions)
    @testset "init_fn(): Type and dimensions check" begin
        msg = "The `init_fn()` function should return a `valid_actions` array of size " *
              "(num_actions, num_envs), and of type `Bool`."
        @assert size(valid_actions)[2] == B && eltype(init_fn_out.valid_actions) == Bool msg
        @test true

        size_logit_prior = size(init_fn_out.logit_prior)
        size_policy_prior = size(init_fn_out.policy_prior)

        msg = "The `logit_prior` and `policy_prior` arrays of `init_fn()` should have " *
              "the same dimensions, equal to (num_actions, num_envs)."
        @assert size_logit_prior == size_policy_prior == (A, B) msg
        @test true

        msg = "The `init_fn()` function should return a `logit_prior` array of size " *
              "(num_actions, num_envs), and of type `Float32`."
        @assert eltype(init_fn_out.logit_prior) == Float32 msg
        @test true

        msg = "The `init_fn()` function should return a `policy_prior` array of size " *
              "(num_actions, num_envs), and of type `Float32`."
        @assert eltype(init_fn_out.policy_prior) == Float32 msg
        @test true

        length_value_prior = length(init_fn_out.value_prior)
        msg = "The `init_fn()` function should return a `value_prior` vector of length " *
              "(num_envs,), and of type `Float32`."
        @assert (length_value_prior == B && eltype(init_fn_out.value_prior) == Float32) msg
        @test true
    end

    # choose the first available action for every environment -- raise error if none
    aids = zeros(Int16, B)
    valid_actions_cpu = Array(valid_actions)
    for bid in 1:B
        if any(valid_actions_cpu[:, bid])
            aids[bid] = findfirst(valid_actions_cpu[:, bid])
        else
            throw(ArgumentError("The environment $bid has no valid actions."))
        end
    end
    aids = DeviceArray(device)(aids)

    # transition_fn()
    transition_fn_out = oracle.transition_fn(envs, aids)

    @testset "transition_fn(): Named-tuple check" begin
        exp = (:internal_states, :rewards, :terminal, :valid_actions, :player_switched,
               :logit_prior, :policy_prior, :value_prior)
        msg = "The `EnvOracle`'s `transition_fn()` function should returned a " *
              "named-tuple with the following fields: internal_states, rewards, " *
              "terminal, valid_actions, player_switched, logit_prior, policy_prior, " *
              "value_prior."
        @assert check_keys(keys(transition_fn_out), exp) msg
        @test true
    end

    @testset "transition_fn(): Type and dimensions check" begin
        rewards = transition_fn_out.rewards
        msg = "The `transition_fn()` function should return a `rewards` vector of length " *
              "(num_envs,), and of type `Float32`."
        @assert length(rewards) == B && eltype(rewards) == Float32 msg
        @test true

        terminal = transition_fn_out.terminal
        msg = "The `transition_fn()` function should return a `terminal` vector of " *
              "length (num_envs,), and of type `Bool`."
        @assert length(terminal) == B && eltype(terminal) == Bool msg
        @test true

        valid_actions = transition_fn_out.valid_actions
        msg = "The `transition_fn()` function should return a `valid_actions` array of " *
              "size (num_actions, num_envs), and of type `Bool`."
        @assert (size(valid_actions) == (A, B) && eltype(valid_actions) == Bool) msg
        @test true

        player_switched = transition_fn_out.player_switched
        msg = "The `transition_fn()` function should return a `player_switched` vector " *
              "of length (num_envs,), and of type `Bool`."
        @assert length(player_switched) == B && eltype(player_switched) == Bool msg
        @test true

        logit_prior = transition_fn_out.logit_prior
        policy_prior = transition_fn_out.policy_prior
        msg = "The `logit_prior` and `policy_prior` arrays of `transition_fn()` should " *
              "have the same dimensions, equal to (num_actions, num_envs)."
        @assert size(logit_prior) == size(policy_prior) == (A, B) msg
        @test true

        msg = "The `transition_fn()` function should return a `logit_prior` array of " *
              "size (num_actions, num_envs), and of type `Float32`."
        @assert eltype(transition_fn_out.logit_prior) == Float32 msg
        @test true

        msg = "The `transition_fn()` function should return a `policy_prior` array of " *
              "size (num_actions, num_envs), and of type `Float32`."
        @assert eltype(transition_fn_out.policy_prior) == Float32 msg
        @test true

        value_prior = transition_fn_out.value_prior
        msg = "The `transition_fn()` function should return a `value_prior` vector of " *
              "length (num_envs,), and of type `Float32`."
        @assert length(value_prior) == B && eltype(value_prior) == Float32 msg
        @test true
    end
end

function test_uniform_env_oracle()
    @testset "Single-Player Env: BitwiseRandomWalk1D" begin
        envs = [
            BitwiseRandomWalk1DEnv(),
            TestEnvs.bitwise_random_walk_losing(),
            TestEnvs.bitwise_random_walk_winning()
        ]
        oracle = uniform_env_oracle()

        @testset "CPU" begin
            envs_arr = Array(envs)
            _check_oracle(oracle, envs_arr)
            @test true
        end

        CUDA.functional() && @testset "GPU" begin
            envs_arr = CuArray(envs)
            _check_oracle(oracle, envs_arr)
            @test true
        end
    end

    @testset "Two-Player Env: BitwiseTicTacToe" begin
        envs = [
            BitwiseTicTacToeEnv(),
            TestEnvs.bitwise_tictactoe_draw(),
            TestEnvs.bitwise_tictactoe_winning()
        ]
        oracle = uniform_env_oracle()

        @testset "CPU" begin
            envs_arr = Array(envs)
            _check_oracle(oracle, envs_arr)
            @test true
        end

        CUDA.functional() && @testset "GPU" begin
            envs_arr = CuArray(envs)
            _check_oracle(oracle, envs_arr)
            @test true
        end
    end
end

function test_neural_network_env_oracle()
    hp = SimpleNetHP(width=256, depth_common=3)

    @testset "Single-Player Env: BitwiseRandomWalk1D" begin
        rw1d_state_size = BatchedEnvs.state_size(BitwiseRandomWalk1DEnv)
        rw1d_na = BatchedEnvs.num_actions(BitwiseRandomWalk1DEnv)
        nn = SimpleNet(rw1d_state_size..., rw1d_na, hp)

        envs = [
            BitwiseRandomWalk1DEnv(),
            TestEnvs.bitwise_random_walk_losing(),
            TestEnvs.bitwise_random_walk_winning()
        ]

        @testset "CPU" begin
            envs_arr = Array(envs)
            oracle = neural_network_env_oracle(; nn=nn)
            _check_oracle(oracle, envs_arr)
            @test true
        end

        CUDA.functional() && @testset "GPU" begin
            envs_arr = CuArray(envs)
            oracle = neural_network_env_oracle(; nn=gpu(nn))
            _check_oracle(oracle, envs_arr)
            @test true
        end
    end

    @testset "Two-Player Env: BitwiseTicTacToe" begin
        ttt_state_size = BatchedEnvs.state_size(BitwiseTicTacToeEnv)
        ttt_na = BatchedEnvs.num_actions(BitwiseTicTacToeEnv)
        nn = SimpleNet(ttt_state_size..., ttt_na, hp)

        envs = [
            BitwiseTicTacToeEnv(),
            TestEnvs.bitwise_tictactoe_draw(),
            TestEnvs.bitwise_tictactoe_winning()
        ]

        @testset "CPU" begin
            envs_arr = Array(envs)
            oracle = neural_network_env_oracle(; nn=nn)
            _check_oracle(oracle, envs_arr)
            @test true
        end

        CUDA.functional() && @testset "GPU" begin
            envs_arr = CuArray(envs)
            oracle = neural_network_env_oracle(; nn=gpu(nn))
            _check_oracle(oracle, envs_arr)
            @test true
        end
    end
end

end
