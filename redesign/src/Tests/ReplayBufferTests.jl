module ReplayBufferTests

using Random: MersenneTwister
using Test

using ...ReplayBuffers
using .....Util.Devices


export run_replay_buffer_tests


function run_replay_buffer_tests()
    @testset "Sample" begin
        @testset "struct creation" test_sample_creation()
    end
    @testset "EpisodeBuffer" begin
        @testset "struct creation" test_episode_buffer_creation()
        @testset "data saving" test_episode_buffer_data_saving()
        # @testset "reward propagation" test_episode_buffer_propagate_reward()
        @testset "value function computation" test_episode_buffer_compute_value_functions()
        @testset "empty" test_episode_buffer_empty()
    end
    @testset "ReplayBuffer" begin
        @testset "add" test_replay_buffer_add()
        @testset "data conversion to arrays" test_replay_buffer_to_array()
    end
end

function test_sample_creation()
    @testset "default constructor" begin
        state = zeros(Float32, 4, 2, 7)
        action = Int16(4)
        reward = 0.0f0
        switched = true

        Sample(state, action, reward, switched)
        @test true
    end
    @testset "custom constructor" begin
        @testset "state size single dimension" begin
            dims = (69,)
            Sample(dims)
            @test true
        end
        @testset "state size multiple dimensions" begin
            dims = (80, 80, 3)
            Sample(dims)
            @test true
        end
    end
end

function test_episode_buffer_creation()
    num_envs = 4
    EpisodeBuffer(num_envs)
    @test true
end

function test_episode_buffer_data_saving()
    num_envs = 256
    rng = MersenneTwister(42)
    state_size = (80, 80, 3)
    ep_buff = EpisodeBuffer(num_envs)

    batch_states = [rand(rng, Float32, state_size...) for _ in 1:num_envs]
    batch_actions = [rand(rng, Int16) for _ in 1:num_envs]
    batch_rewards = [rand(rng, Float32) for _ in 1:num_envs]
    batch_switches = [rand(rng, Bool) for _ in 1:num_envs]

    @testset "batch saving" begin
        save!(ep_buff, batch_states, batch_actions, batch_rewards, batch_switches)
        @test true
    end

    @testset "data saving correctness" begin
        map(1:num_envs) do env_id
            @test length(ep_buff[env_id]) == 1
            @test ep_buff[env_id][1].state == batch_states[env_id]
            @test ep_buff[env_id][1].action == batch_actions[env_id]
            @test ep_buff[env_id][1].reward == batch_rewards[env_id]
            @test ep_buff[env_id][1].switched == batch_switches[env_id]
        end
    end
end

function test_episode_buffer_compute_value_functions()
    num_envs = 3
    rng = MersenneTwister(13)
    state_size = (27,)
    ep_buff = EpisodeBuffer(num_envs)

    env1_rewards = Float32.([1, 2, 3, -1, -1, 2, 0])
    env1_switches = [true, false, true, false, true, true, true]
    env1_labels = Float32.([-2, -3, 5, 2, -3, 2, 0])
    env1_γ = 1f0

    env2_rewards = Float32.([0, 0, 0, 0, -1])
    env2_switches = [true, true, true, true, true]
    env2_labels = Float32.([-1 * 0.99f0 ^ 4, 1 * 0.99f0 ^ 3, -1 * 0.99f0 ^ 2, 1 * 0.99f0, -1])
    env2_γ = 0.99f0

    env3_rewards = Float32.([0.05, 0, 0.2, 0.3, 0.07, 0.0, -0.3, 0.1])
    env3_switches = [false, false, false, false, false, false, false, false]
    env3_labels = Float32.([0.42, 0.37, 0.37, 0.17, -0.13, -0.2, -0.2, 0.1])
    env3_γ = 1f0

    env_rewards = [env1_rewards, env2_rewards, env3_rewards]
    env_switches = [env1_switches, env2_switches, env3_switches]
    env_labels = [env1_labels, env2_labels, env3_labels]
    env_γ = [env1_γ, env2_γ, env3_γ]

    @testset "manually saving to episode buffer" begin
        map(1:num_envs) do env_id
            for step in 1:length(env_rewards[env_id])
                state = rand(rng, Float32, state_size...)
                action = rand(rng, Int16)
                reward = env_rewards[env_id][step]
                switched = env_switches[env_id][step]

                sample = Sample(state, action, reward, switched)
                push!(ep_buff[env_id], sample)
            end
            @test true
        end
    end

    @testset "value function (rewards to go) computation" begin
        map(1:num_envs) do env_id
            compute_value_functions!(ep_buff, env_id, env_γ[env_id])
            @test true
        end
    end

    @testset "value function computation correctness" begin
        map(1:num_envs) do env_id
            pred_labels = map(idx -> ep_buff[env_id][idx].reward, 1:length(ep_buff[env_id]))
            @test isapprox(pred_labels, env_labels[env_id], atol=1e-7)
        end
    end
end

function test_episode_buffer_empty()
    num_envs = 17
    rng = MersenneTwister(0)
    state_size = (126,)
    episode_lengths = [rand(rng, 1:50) for _ in 1:num_envs]
    ep_buff = EpisodeBuffer(num_envs)

    function is_empty(_is_empty = true)
        empties = map(env_id -> length(ep_buff[env_id]), 1:num_envs)
        @test all(empties .== 0) == _is_empty
    end

    @testset "empty initially" is_empty(true)

    map(1:num_envs) do env_id
        for step in 1:episode_lengths[env_id]
            state = rand(rng, Float32, state_size...)
            action = rand(rng, Int16)
            reward = (step == episode_lengths[env_id]) ? 0f0 : Float32(rand(rng, -1:1))
            switched = rand(rng, Bool)

            sample = Sample(state, action, reward, switched)
            push!(ep_buff[env_id], sample)
        end
    end

    @testset "not empty after saving" is_empty(false)

    @testset "emptying" begin
        map(1:num_envs) do env_id
            empty_env_in_buffer!(ep_buff, env_id)
            @test true
        end
    end

    @testset "empty after emptying" is_empty(true)
end

function test_replay_buffer_add()
    num_envs = 9
    rng = MersenneTwister(0)
    state_size, num_actions = (28, 28), 10
    max_steps_per_env = 18
    steps = [rand(rng, 1:50) for _ in 1:num_envs]
    ep_buff = EpisodeBuffer(num_envs)
    rp_buff = ReplayBuffer(num_envs, max_steps_per_env, state_size, num_actions)

    @testset "filling buffers" begin
        map(1:num_envs) do env_id
            for step in 1:steps[env_id]
                state = rand(rng, Float32, state_size...)
                action = rand(rng, Int16)
                reward = (step == steps[env_id]) ? 0f0 : Float32(rand(rng, -1:1))
                switched = rand(rng, Bool)

                sample = Sample(state, action, reward, switched)
                push!(ep_buff[env_id], sample)

                done = rand(rng, Float32) < 0.1
                if done
                    compute_value_functions!(ep_buff, env_id, 1f0)
                    add!(rp_buff, ep_buff, env_id)
                    empty_env_in_buffer!(ep_buff, env_id)
                    @test true
                end
            end
        end
    end

    @testset "testing addition correctness" begin
        map(1:num_envs) do env_id
            common_len = length(rp_buff.states[env_id])
            @test common_len <= max_steps_per_env
            @test length(rp_buff.actions[env_id]) == common_len
            @test length(rp_buff.rewards[env_id]) == common_len
        end
    end
end

function test_replay_buffer_to_array()
    num_envs = 13
    rng = MersenneTwister(9)
    state_size, num_actions = (80, 80, 3), 4
    max_steps_per_env = 50
    total_steps = 222

    ep_buff = EpisodeBuffer(num_envs)
    rp_buff = ReplayBuffer(num_envs, max_steps_per_env, state_size, num_actions)

    @testset "filling buffers" begin
        for _ in 1:total_steps
            batch_states = [rand(rng, Float32, state_size...) for _ in 1:num_envs]
            batch_actions = [Int16(rand(rng, 1:num_actions)) for _ in 1:num_envs]
            batch_rewards = [rand(rng, Float32) for _ in 1:num_envs]
            batch_switches = [rand(rng, Bool) for _ in 1:num_envs]
            batch_dones = rand(rng, Bool, num_envs) .< 0.05

            save!(ep_buff, batch_states, batch_actions, batch_rewards, batch_switches)
            @test true

            map(1:num_envs) do env_id
                if batch_dones[env_id]
                    compute_value_functions!(ep_buff, env_id, 1f0)
                    add!(rp_buff, ep_buff, env_id)
                    empty_env_in_buffer!(ep_buff, env_id)
                    @test true
                end
            end
        end
    end

    @testset "data conversion correctness" begin
        num_training_samples = sum(map(env_id -> length(rp_buff.states[env_id]), 1:num_envs))

        @testset "GPU-check" begin
            to_array(rp_buff, GPU())
            @test true
        end

        all_states, all_actions, all_rewards = to_array(rp_buff, CPU())
        @test size(all_states)[end] == num_training_samples
        @test size(all_actions)[end] == num_training_samples
        @test size(all_rewards)[end] == num_training_samples
    end
end

end
