module BitwiseRandomWalk1DEvalFns

using Flux
using Distributions
using Logging
using Plots
using Random

using ..BitwiseRandomWalk1D
using ....BatchedEnvs
using ....BatchedMctsUtilities
using ....Network
using ....LoggingUtilities
using ....Util.Devices


export get_nn_evaluation_fn, plot_metrics


function _get_nn_policy(cpu_nn, env)
    state = BatchedEnvs.vectorize_state(env)
    state = Flux.unsqueeze(state, length(size(state)) + 1)
    _, p = forward(cpu_nn, state, true)
    return p[:, 1]
end

function get_nn_evaluation_fn(metrics)

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
        push!(metrics, (avg_steps, avg_win_rate, avg_right_prob))
    end

    return evaluate_nn
end

function plot_metrics(save_dir, timestamps, metrics)
    !isdir(save_dir) && mkpath(save_dir)

    avg_steps = [metrics[1] for metrics in metrics]
    avg_win_rates = [metrics[2] for metrics in metrics]
    avg_right_probs = [metrics[3] for metrics in metrics]

    l = @layout [a ; b ; c]

    p1 = plot(timestamps, avg_steps, label="Average episode length", ylims=(0, 20),
              title="Metrics", linewidth=2, legend=:best, show=false)
    p2 = plot(timestamps, avg_win_rates, label="Win rate over 10 episodes",
              ylims=(0, 1.01), linewidth=2, legend=:best, show=false)
    p3 = plot(timestamps, avg_right_probs, label="Average optimal policy probability",
              xlabel="\ntraining time (s)", ylims=(0, 1.01), linewidth=2,
              legend=:bottomright, show=false)

    plot_size = (Int(floor(750 / Base.MathConstants.golden)), 750)
    p = plot(p1, p2, p3, layout=l, size=plot_size, margin=(6, :mm), show=false)
    savefig(p, joinpath(save_dir, "metrics.png"))

    return nothing
end

end
