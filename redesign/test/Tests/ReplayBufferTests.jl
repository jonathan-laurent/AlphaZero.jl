module ReplayBufferTests

using EllipsisNotation
using Random: MersenneTwister
using Test

using ...ReplayBuffers
using .....Util.Devices


export run_replay_buffer_tests


function run_replay_buffer_tests()
    @testset "EpisodeBuffer" begin
        @testset "struct creation" test_episode_buffer_creation()
        @testset "data saving" test_episode_buffer_data_saving()
        @testset "value function computation" test_episode_buffer_compute_value_functions()
        @testset "empty" test_episode_buffer_empty()
    end
    @testset "ReplayBuffer" begin
        @testset "add" test_replay_buffer_add()
        @testset "data conversion to arrays" test_replay_buffer_to_array()
    end
end

function test_episode_buffer_creation()
    num_envs = 4
    state_size = (80, 80, 3)
    initial_horizon = 10
    EpisodeBuffer(num_envs, state_size, initial_horizon)
    @test true
end

function test_episode_buffer_data_saving()
    num_envs = 3
    state_size = (80, 80, 3)
    ep_buff = EpisodeBuffer(num_envs, state_size)

    rng = MersenneTwister(42)
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
            @test ep_buff.ep_lengths[env_id] == 1
            @test ep_buff.states[.., 1, env_id] == batch_states[env_id]
            @test ep_buff.actions[1, env_id] == batch_actions[env_id]
            @test ep_buff.rewards[1, env_id] == batch_rewards[env_id]
            @test ep_buff.switches[1, env_id] == batch_switches[env_id]
        end
    end
end

function test_episode_buffer_compute_value_functions()
    num_envs = 3
    state_size = (27,)
    ep_buff = EpisodeBuffer(num_envs, state_size)

    env1_rewards = Float32.([1, 2, 3, -1, -1, 2, 0])
    env1_switches = [true, false, true, false, true, true, true]
    env1_labels = Float32.([-2, -3, 5, 2, -3, 2, 0])
    env1_γ = 1f0

    env2_rewards = Float32.([0, 0, 0, 0, -1])
    env2_switches = [true, true, true, true, true]
    env2_labels = [-1 * 0.99f0 ^ 4, 1 * 0.99f0 ^ 3, -1 * 0.99f0 ^ 2, 1 * 0.99f0, -1f0]
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
                ep_buff.rewards[step, env_id] = env_rewards[env_id][step]
                ep_buff.switches[step, env_id] = env_switches[env_id][step]
                ep_buff.ep_lengths[env_id] += 1
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
            ep_len = ep_buff.ep_lengths[env_id]
            pred_labels = ep_buff.rewards[1:ep_len, env_id]
            @test isapprox(pred_labels, env_labels[env_id], atol=1e-7)
        end
    end
end

function test_episode_buffer_empty()
    num_envs = 4
    rng = MersenneTwister(0)
    state_size = (84,)
    total_steps = 50
    ep_buff = EpisodeBuffer(num_envs, state_size)

    function is_empty(_is_empty = true)
        empties = map(env_id -> ep_buff.ep_lengths[env_id], 1:num_envs)
        @test all(empties .== 0) == _is_empty
    end

    @testset "empty initially" is_empty(true)

    for _ in 1:num_envs
        states = [rand(rng, Float32, state_size...) for _ in 1:total_steps]
        actions = [rand(rng, Int16) for _ in 1:total_steps]
        rewards = [rand(rng, Float32) for _ in 1:total_steps]
        switches = [rand(rng, Bool) for _ in 1:total_steps]

        save!(ep_buff, states, actions, rewards, switches)
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
    num_envs = 3
    max_size_per_env = 4
    max_size = num_envs  * max_size_per_env
    total_steps_per_env = max_size_per_env + 3
    state_size, num_actions = (3, 3), 10
    ep_buff = EpisodeBuffer(num_envs, state_size)
    rp_buff = ReplayBuffer(max_size, state_size, num_actions)

    rng = MersenneTwister(0)
    states = rand(rng, Float32, state_size..., total_steps_per_env, num_envs)
    actions = rand(rng, Int16, total_steps_per_env, num_envs)
    rewards = rand(rng, Float32, total_steps_per_env, num_envs)
    switches = rand(rng, Bool, total_steps_per_env, num_envs)

    @testset "filling buffers" begin
        for step in 1:max_size_per_env
            svec = [states[.., step, env_id] for env_id in 1:num_envs]
            a = actions[step, :]
            r = rewards[step, :]
            sw = switches[step, :]
            save!(ep_buff, svec, a, r, sw)
        end

        map(1:num_envs) do env_id
            add!(rp_buff, ep_buff, env_id)
            empty_env_in_buffer!(ep_buff, env_id)
        end

        @test true
    end

    @testset "testing addition correctness" begin
        @test length(rp_buff) == max_size
        map(1:num_envs) do env_id
            rp_range = ((env_id - 1) * max_size_per_env + 1):(env_id * max_size_per_env)
            @test rp_buff.states[.., rp_range] == states[.., 1:max_size_per_env, env_id]
            @test rp_buff.actions[rp_range] == actions[1:max_size_per_env, env_id]
            @test rp_buff.values[1, rp_range] == rewards[1:max_size_per_env, env_id]
        end
    end

    @testset "overwriting old data" begin
        for step in (max_size_per_env + 1):total_steps_per_env
            svec = [states[.., step, env_id] for env_id in 1:num_envs]
            a = actions[step, :]
            r = rewards[step, :]
            sw = switches[step, :]
            save!(ep_buff, svec, a, r, sw)
        end

        map(1:num_envs) do env_id
            add!(rp_buff, ep_buff, env_id)
            empty_env_in_buffer!(ep_buff, env_id)
        end

        @test true
    end

    @testset "testing overwriting correctness" begin
        @test length(rp_buff) == max_size
        num_new_steps = total_steps_per_env - max_size_per_env

        @testset "overwriting of new data" begin
            map(1:num_envs) do env_id
                rp_range = ((env_id - 1) * num_new_steps + 1):(env_id * num_new_steps)
                data_range = (max_size_per_env + 1):total_steps_per_env

                @test rp_buff.states[.., rp_range] == states[.., data_range, env_id]
                @test rp_buff.actions[rp_range] == actions[data_range, env_id]
                @test rp_buff.values[1, rp_range] == rewards[data_range, env_id]
            end
        end

        @testset "appropriate old data hasn't been removed" begin
            new_steps = (num_envs * num_new_steps)
            last_idx = length(rp_buff)
            for env_id in reverse(1:num_envs)

                num_old_data = min(max_size_per_env, last_idx - new_steps)
                rp_range = (last_idx - num_old_data + 1):last_idx
                data_range = (max_size_per_env - num_old_data + 1):max_size_per_env

                @test rp_buff.states[.., rp_range] == states[.., data_range, env_id]
                @test rp_buff.actions[rp_range] == actions[data_range, env_id]
                @test rp_buff.values[1, rp_range] == rewards[data_range, env_id]

                last_idx -= num_old_data
                (last_idx == new_steps) && break
            end
        end
    end
end

function test_replay_buffer_to_array()
    num_envs = 3
    rng = MersenneTwister(9)
    state_size, num_actions = (80, 80, 3), 4
    steps_per_env = 50
    max_size = num_envs * 222

    ep_buff = EpisodeBuffer(num_envs, state_size)
    rp_buff = ReplayBuffer(max_size, state_size, num_actions)

    @testset "filling buffers" begin
        for _ in 1:steps_per_env
            batch_states = [rand(rng, Float32, state_size...) for _ in 1:num_envs]
            batch_actions = [Int16(rand(rng, 1:num_actions)) for _ in 1:num_envs]
            batch_rewards = [rand(rng, Float32) for _ in 1:num_envs]
            batch_switches = [rand(rng, Bool) for _ in 1:num_envs]
            batch_dones = rand(rng, Bool, num_envs) .< 0.05

            save!(ep_buff, batch_states, batch_actions, batch_rewards, batch_switches)

            map(1:num_envs) do env_id
                if batch_dones[env_id]
                    compute_value_functions!(ep_buff, env_id, 1f0)
                    add!(rp_buff, ep_buff, env_id)
                    empty_env_in_buffer!(ep_buff, env_id)
                end
            end
        end
        @test true
    end

    @testset "data conversion correctness" begin
        num_training_samples = length(rp_buff)

        all_states, all_actions, all_values = to_array(rp_buff)
        @test size(all_states)[end] == num_training_samples
        @test size(all_actions)[end] == num_training_samples
        @test size(all_values)[end] == num_training_samples
    end
end

end
