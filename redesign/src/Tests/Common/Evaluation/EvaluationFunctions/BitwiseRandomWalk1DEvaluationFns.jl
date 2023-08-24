module BitwiseRandomWalk1DEvalFns

using Flux
using Distributions
using Logging
using Random

using ..BitwiseRandomWalk1D
using ....BatchedEnvs
using ....BatchedMctsUtilities
using ....Network
using ....LoggingUtilities
using ....Util.Devices


export evaluate_nn


function _get_nn_policy(cpu_nn, env)
    state = BatchedEnvs.vectorize_state(env)
    state = Flux.unsqueeze(state, length(size(state)) + 1)
    _, p = forward(cpu_nn, state)
    invalid_actions = .!get_valid_actions([env])[:, 1]
    p[invalid_actions, 1] .= -Inf
    return Flux.softmax(p[:, 1])
end

function evaluate_nn(loggers, nn, _, _)
    rng = Random.MersenneTwister(0)
    cpu_nn = Flux.cpu(nn)
    num_wins, total_steps, right_probs = 0, 0, 0f0
    for _ in 1:10
        env = BitwiseRandomWalk1DEnv()

        step, done, info, episode_right_probs = 0, false, nothing, 0f0
        while step < 50 && !done
            step += 1
            policy = _get_nn_policy(cpu_nn, env)
            episode_right_probs += policy[2]
            action = rand(rng, Distributions.Categorical(policy))
            env, info = BatchedEnvs.act(env, action)
            done = BatchedEnvs.terminated(env)
        end

        (info.reward > 0) && (num_wins += 1)
        total_steps += step
        avg_episode_right_prob = episode_right_probs / step
        right_probs += avg_episode_right_prob
    end

    avg_steps = total_steps / 10
    avg_win_rate = num_wins / 10
    avg_right_prob = right_probs / 10

    with_logger(loggers["tb"]) do
        @info "eval" nn_avg_steps=avg_steps log_step_increment=0
        @info "eval" nn_avg_win_rate=avg_win_rate log_step_increment=0
        @info "eval" nn_avg_optimal_policy_prob=avg_right_prob log_step_increment=0
    end
end

end
