module BitwiseTicTacToeEvalFns

using Flux
using Logging
using Plots
using Random

using ..BitwiseTicTacToe
using ..BitwiseTicTacToeHeuristic
using ....BatchedEnvs
using ....BatchedMcts
using ....BatchedMctsUtilities
using ....Evaluation
using ....Network
using ....LoggingUtilities
using ....Minimax
using ....TrainUtilities
using ....Util.Devices


export get_alphazero_vs_random_eval_fn, get_nn_vs_random_eval_fn
export get_alphazero_vs_minimax_eval_fn, get_nn_vs_minimax_eval_fn
export get_tictactoe_benchmark_fns
export plot_metrics


const MCTS = BatchedMcts


function act_minimax(single_env_vec, kwargs)
    env = Array(single_env_vec)[1]
    depth = kwargs["iteration"]
    is_maximizing = env.curplayer == BitwiseTicTacToe.CROSS
    return minimax(env; depth, eval_fn=tictactoe_eval_fn, is_maximizing)
end

function write_episode_action_sequence(loggers, episode, action_sequence, p1_reward, pname)
    episode_str = lpad(episode, 2, "0")
    msg = "\tEpisode $episode_str:\tAction sequence: $action_sequence => " *
           "$pname reward: $(p1_reward)\n"
    write_msg(loggers["eval"], msg)
end

function get_alphazero_vs_random_eval_fn(random_kwargs, metrics)

    function evaluate_alphazero_vs_random(loggers, nn, _, _)
        az_args = (init_mcts_config(CPU(), Flux.cpu(nn), random_kwargs["config"]),)
        az_wins, az_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of AlphaZero vs random policy")
        for iter in 1:random_kwargs["num_bilateral_rounds"]
            env = BitwiseTicTacToeEnv()

            # az starts
            r, seq = play_round(env, mcts_move, az_args, random_move, random_kwargs, true)
            write_episode_action_sequence(loggers, iter, seq, r, "alphazero")
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_wins += 1) : (az_losses += 1))

            # random starts
            r, seq = play_round(env, mcts_move, az_args, random_move, random_kwargs, false)
            write_episode_action_sequence(loggers, iter, seq, r, "alphazero")
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_wins += 1) : (az_losses += 1))
        end

        win_rate = az_wins / (2 * random_kwargs["num_bilateral_rounds"])
        loss_rate = az_losses / (2 * random_kwargs["num_bilateral_rounds"])
        draw_rate = draws / (2 * random_kwargs["num_bilateral_rounds"])
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_az_vs_random=win_rate log_step_increment=0
            @info "eval" loss_rate_az_vs_random=loss_rate log_step_increment=0
            @info "eval" draw_rate_az_vs_random=draw_rate log_step_increment=0
        end
        push!(metrics["az"]["random"], (win_rate, loss_rate, draw_rate))
    end

    return evaluate_alphazero_vs_random
end

function get_nn_vs_random_eval_fn(random_kwargs, metrics)

    function evaluate_nn_vs_random(loggers, nn, _, _)
        cpu_nn = nn |> Flux.cpu
        nn_wins, nn_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of NN vs random policy")
        for iter in 1:random_kwargs["num_bilateral_rounds"]
            env = BitwiseTicTacToeEnv()

            # nn starts
            r, seq = play_round(env, nn_move, (cpu_nn,), random_move, random_kwargs, true)
            write_episode_action_sequence(loggers, iter, seq, r, "nn")
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_wins += 1) : (nn_losses += 1))

            # random starts
            r, seq = play_round(env, nn_move, (cpu_nn,), random_move, random_kwargs, false)
            write_episode_action_sequence(loggers, iter, seq, r, "nn")
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_wins += 1) : (nn_losses += 1))
        end

        win_rate = nn_wins / (2 * random_kwargs["num_bilateral_rounds"])
        loss_rate = nn_losses / (2 * random_kwargs["num_bilateral_rounds"])
        draw_rate = draws / (2 * random_kwargs["num_bilateral_rounds"])
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_nn_vs_random=win_rate log_step_increment=0
            @info "eval" loss_rate_nn_vs_random=loss_rate log_step_increment=0
            @info "eval" draw_rate_nn_vs_random=draw_rate log_step_increment=0
        end
        push!(metrics["nn"]["random"], (win_rate, loss_rate, draw_rate))
    end

    return evaluate_nn_vs_random
end

function get_alphazero_vs_minimax_eval_fn(minimax_kwargs, metrics)

    function evaluate_alphazero_vs_minimax(loggers, nn, _, _)
        az_args = (init_mcts_config(CPU(), Flux.cpu(nn), minimax_kwargs["config"]),)
        az_wins, az_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of AlphaZero vs minimax policy")
        for iter in 1:minimax_kwargs["num_bilateral_rounds"]
            env = BitwiseTicTacToeEnv()
            minimax_kwargs["iteration"] = iter

            # az starts
            r, seq = play_round(env, mcts_move, az_args, act_minimax, minimax_kwargs, true)
            write_episode_action_sequence(loggers, iter, seq, r, "alphazero")
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_wins += 1) : (az_losses += 1))

            # minimax starts
            r, seq = play_round(env, mcts_move, az_args, act_minimax, minimax_kwargs, false)
            write_episode_action_sequence(loggers, iter, seq, r, "alphazero")
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_wins += 1) : (az_losses += 1))
        end

        win_rate = az_wins / (2 * minimax_kwargs["num_bilateral_rounds"])
        loss_rate = az_losses / (2 * minimax_kwargs["num_bilateral_rounds"])
        draw_rate = draws / (2 * minimax_kwargs["num_bilateral_rounds"])
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_az_vs_minimax=win_rate log_step_increment=0
            @info "eval" loss_rate_az_vs_minimax=loss_rate log_step_increment=0
            @info "eval" draw_rate_az_vs_minimax=draw_rate log_step_increment=0
        end
        push!(metrics["az"]["minimax"], (win_rate, loss_rate, draw_rate))
    end

    return evaluate_alphazero_vs_minimax
end

function get_nn_vs_minimax_eval_fn(minimax_kwargs, metrics)

    function evaluate_nn_vs_minimax(loggers, nn, _, _)
        cpu_nn = nn |> Flux.cpu
        nn_wins, nn_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of NN vs minimax policy")
        for iter in 1:minimax_kwargs["num_bilateral_rounds"]
            env = BitwiseTicTacToeEnv()
            minimax_kwargs["iteration"] = iter

            # nn starts
            r, seq = play_round(env, nn_move, (cpu_nn,), act_minimax, minimax_kwargs, true)
            write_episode_action_sequence(loggers, iter, seq, r, "nn")
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_wins += 1) : (nn_losses += 1))

            # minimax starts
            r, seq = play_round(env, nn_move, (cpu_nn,), act_minimax, minimax_kwargs, false)
            write_episode_action_sequence(loggers, iter, seq, r, "nn")
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_wins += 1) : (nn_losses += 1))
        end

        win_rate = nn_wins / (2 * minimax_kwargs["num_bilateral_rounds"])
        loss_rate = nn_losses / (2 * minimax_kwargs["num_bilateral_rounds"])
        draw_rate = draws / (2 * minimax_kwargs["num_bilateral_rounds"])
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_nn_vs_minimax=win_rate log_step_increment=0
            @info "eval" loss_rate_nn_vs_minimax=loss_rate log_step_increment=0
            @info "eval" draw_rate_nn_vs_minimax=draw_rate log_step_increment=0
        end
        push!(metrics["nn"]["minimax"], (win_rate, loss_rate, draw_rate))
    end

    return evaluate_nn_vs_minimax
end

function get_tictactoe_benchmark_fns(kwargs, metrics)
    _, optimal_policy, possible_envs = minimax_compute_optimal_actions(
        BitwiseTicTacToeEnv();
        depth=10
    )

    num_valid_actions(env) = sum(BatchedEnvs.valid_action(env, a) for a in 1:9)
    possible_envs = [env for env in possible_envs if num_valid_actions(env) > 1]
    total_states = length(possible_envs)

    states = [Array(BatchedEnvs.vectorize_state(env)) for env in possible_envs]
    states = hcat(states...)
    optimal_actions = [optimal_policy[string(env)] for env in possible_envs]

    mcts_losses_logger = LoggingUtilities.init_file_logger("mcts_fails.log"; overwrite=true)
    nn_losses_logger = LoggingUtilities.init_file_logger("nn_fails.log"; overwrite=true)

    function add_info_to_env_str(env, a, opt, idx)
        env_encoding = string(env)
        idx = lpad(idx, length(string(total_states)), "0")
        rows = split(env_encoding, "\n")
        curplayer_row, tictactoe_rows = rows[1], rows[3:end]
        info = "Action chosen: $a. Optimal Actions: $opt"
        fixed = "\n$curplayer_row\n" *
                "             " * tictactoe_rows[1] * "\n" *
                "State $idx:  " * tictactoe_rows[2] * "\t$info\n" *
                "             " * tictactoe_rows[3] * "\n"
        return fixed
    end

    function benchmark_az_fn(loggers, nn, _, step)
        envs = DeviceArray(kwargs["device"])(possible_envs)
        device_nn = (kwargs["device"] == GPU()) ? Flux.gpu(nn) : Flux.cpu(nn)
        mcts_config = init_mcts_config(kwargs["device"], device_nn, kwargs["config"])

        tree = MCTS.explore(mcts_config, envs)
        policy = Array(MCTS.evaluation_policy(tree, mcts_config))
        children_visits = Array(MCTS.get_root_children_visits(tree, mcts_config))

        log_msg(mcts_losses_logger, "Accuracy evaluation at step $step")

        false_positives = zeros(Int, 9)
        az_correct = 0
        for i in 1:total_states

            chosen_action = policy[i]
            if chosen_action in optimal_actions[i]
                az_correct += 1
            else
                false_positives[chosen_action] += 1
                env = possible_envs[i]
                info = add_info_to_env_str(env, chosen_action, optimal_actions[i], i)
                write_msg(mcts_losses_logger, info)
                write_msg(mcts_losses_logger, "Visit counts: $(children_visits[:, i])\n\n")
            end
        end
        log_msg(mcts_losses_logger, "\nFalse positive actions: $false_positives\n")

        az_accuracy = round(az_correct / total_states, digits=4)
        with_logger(loggers["tb"]) do
            @info "eval" az_benchmark_accuracy=az_accuracy log_step_increment=0
        end
        log_msg(loggers["eval"], "AlphaZero Accuracy: $az_accuracy")
        push!(metrics["az"]["benchmark"], az_accuracy)
    end

    function benchmark_nn_fn(loggers, nn, _, step)
        cpu_nn = nn |> Flux.cpu
        _, logits = forward(cpu_nn, states, false)
        valid_actions = get_valid_actions(possible_envs)

        log_msg(nn_losses_logger, "Accuracy evaluation at step $step")

        false_positives = zeros(Int, 9)
        nn_correct = 0
        for i in 1:total_states
            action_mask = typemin(Float32) .* .!valid_actions[:, i]
            masked_env_logits = logits[:, i] .+ action_mask

            chosen_action = argmax(masked_env_logits)
            if chosen_action in optimal_actions[i]
                nn_correct += 1
            else
                false_positives[chosen_action] += 1
                env = possible_envs[i]
                info = add_info_to_env_str(env, chosen_action, optimal_actions[i], i)
                write_msg(nn_losses_logger, info)
                write_msg(nn_losses_logger, "Masked Logits: $masked_env_logits\n\n")
            end
        end
        log_msg(nn_losses_logger, "\nFalse positive actions: $false_positives\n")

        nn_accuracy = round(nn_correct / total_states, digits=4)
        with_logger(loggers["tb"]) do
            @info "eval" nn_benchmark_accuracy=nn_accuracy log_step_increment=0
        end
        log_msg(loggers["eval"], "Raw NN Accuracy: $nn_accuracy")
        push!(metrics["nn"]["benchmark"], nn_accuracy)
    end

    return benchmark_az_fn, benchmark_nn_fn
end

function plot_metrics(save_dir, timestamps, metrics)
    !isdir(save_dir) && mkpath(save_dir)

    num_matches = length(metrics["az"]["random"])
    az_random_wins = [metrics["az"]["random"][i][1] for i in 1:num_matches]
    az_random_losses = [metrics["az"]["random"][i][2] for i in 1:num_matches]
    az_random_draws = [metrics["az"]["random"][i][3] for i in 1:num_matches]

    num_matches = length(metrics["az"]["minimax"])
    az_minimax_wins = [metrics["az"]["minimax"][i][1] for i in 1:num_matches]
    az_minimax_losses = [metrics["az"]["minimax"][i][2] for i in 1:num_matches]
    az_minimax_draws = [metrics["az"]["minimax"][i][3] for i in 1:num_matches]

    num_matches = length(metrics["nn"]["random"])
    nn_random_wins = [metrics["nn"]["random"][i][1] for i in 1:num_matches]
    nn_random_losses = [metrics["nn"]["random"][i][2] for i in 1:num_matches]
    nn_random_draws = [metrics["nn"]["random"][i][3] for i in 1:num_matches]

    num_matches = length(metrics["nn"]["minimax"])
    nn_minimax_wins = [metrics["nn"]["minimax"][i][1] for i in 1:num_matches]
    nn_minimax_losses = [metrics["nn"]["minimax"][i][2] for i in 1:num_matches]
    nn_minimax_draws = [metrics["nn"]["minimax"][i][3] for i in 1:num_matches]

    num_evals = length(metrics["az"]["benchmark"])
    az_benchmark_accuracies = [metrics["az"]["benchmark"][i] for i in 1:num_evals]

    num_evals = length(metrics["nn"]["benchmark"])
    nn_benchmark_accuracies = [metrics["nn"]["benchmark"][i] for i in 1:num_evals]

    l = @layout [a b c ; d e f; _ g _]
    timestamps /= 60

    p1 = plot(timestamps, [az_random_wins, nn_random_wins], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="\nWin Rate vs Random\n", linewidth=2,
              label=nothing, show=false)
    p2 = plot(timestamps, [az_random_losses, nn_random_losses], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="\nLoss Rate vs Random\n", linewidth=2,
              label=nothing, show=false)
    p3 = plot(timestamps, [az_random_draws, nn_random_draws], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="\nDraw Rate vs Random\n", linewidth=2,
              label=nothing, show=false)
    p4 = plot(timestamps, [az_minimax_wins, nn_minimax_wins], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="Win Rate vs Minimax\n", linewidth=2,
              label=nothing, show=false)
    p5 = plot(timestamps, [az_minimax_losses, nn_minimax_losses], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="Loss Rate vs Minimax\n", linewidth=2,
              label=nothing, show=false)
    p6 = plot(timestamps, [az_minimax_draws, nn_minimax_draws], lc=[:auto :orange],
              ylims=(-0.01, 1.01), title="Draw Rate vs Minimax\n", linewidth=2,
              label=nothing, show=false)
    p7 = plot(timestamps, [az_benchmark_accuracies, nn_benchmark_accuracies],
              lc=[:auto :orange], label=["AlphaZero" "Network Only"], ylims=(-0.01, 1.01),
              xlabel="training time (min)\n", title="Benchmark Accuracy\n", linewidth=2,
              legend=:best, legendfont=font(11), show=false)

    plot_size = (1_600, Int(floor(1_600 / Base.MathConstants.golden)))
    p = plot(p1, p2, p3, p4, p5, p6, p7, layout=l, size=plot_size, margin=(6, :mm),
             show=false)
    savefig(p, joinpath(save_dir, "metrics.png"))

    return nothing
end

end
