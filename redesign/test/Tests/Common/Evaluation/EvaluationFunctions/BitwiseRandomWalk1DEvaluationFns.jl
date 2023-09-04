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
    _, p = forward(cpu_nn, state)
    invalid_actions = .!get_valid_actions([env])[:, 1]
    p[invalid_actions, 1] .= -Inf
    return Flux.softmax(p[:, 1])
end

function get_nn_evaluation_fn(global_times, global_metrics)
    start_time = time()

    function evaluate_nn(loggers, nn, _, _)
        bench_start_time = time()
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
        bench_end_time = time()

        # increment start time by the time it took to evaluate the benchmarks
        start_time += bench_end_time - bench_start_time

        eval_time = time() - start_time
        push!(global_times["nn"], eval_time)
        push!(global_metrics["nn"], (avg_steps, avg_win_rate, avg_right_prob))
    end

    return evaluate_nn
end

function plot_metrics(save_dir, global_times, global_metrics)
    !isdir(save_dir) && mkpath(save_dir)

    times = global_times["nn"]
    avg_steps = [metrics[1] for metrics in global_metrics["nn"]]
    avg_win_rates = [metrics[2] for metrics in global_metrics["nn"]]
    avg_right_probs = [metrics[3] for metrics in global_metrics["nn"]]

    l = @layout [a ; b ; c]

    p1 = plot(times, avg_steps, label="Average steps to episode end", ylims=(0, 20),
              title="Metrics\n", linewidth=2, legend=:best, show=false)
    p2 = plot(times, avg_win_rates, label="Average win rate over 10 episodes",
              ylims=(0, 1.01), linewidth=2, legend=:best, show=false)
    p3 = plot(times, avg_right_probs, label="Average optimal policy selection",
              xlabel="\ntraining time (s)", ylims=(0, 1.01), linewidth=2, legend=:best,
              show=false)

    plot_size = (Int(floor(1_000 / Base.MathConstants.golden)), 1_000)
    p = plot(p1, p2, p3, layout=l, size=plot_size, margin=(5, :mm), show=false)
    savefig(p, joinpath(save_dir, "metrics.png"))

    return nothing
end

end
