module BiwtiseTicTacToeEvalFns

using Flux
using Logging
using Random

using ..BitwiseTicTacToe
using ..BitwiseTicTacToeHeuristic
using ....BatchedEnvs
using ....BatchedMcts
using ....BatchedMctsUtilities: Policy
using ....EnvOracles
using ....Evaluation
using ....Network
using ....LoggingUtilities
using ....Minimax
using ....Util.Devices

export get_alphazero_vs_random_eval_fn, get_nn_vs_random_eval_fn
export get_alphazero_vs_minimax_eval_fn, get_nn_vs_minimax_eval_fn
export get_tictactoe_benchmark_fns

const MCTS = BatchedMcts


function act_minimax(single_env_vec, kwargs)
    env = single_env_vec[1]
    depth = kwargs["iteration"]
    is_maximizing = env.curplayer == BitwiseTicTacToe.CROSS
    return deterministic_minimax(env; depth, eval_fn=tictactoe_eval_fn, is_maximizing)
end

function write_episode_action_sequence(loggers, episode, action_sequence, p1_reward, pname)
    episode_str = lpad(episode, 2, "0")
    msg = "\tEpisode $episode_str:\tAction sequence: $action_sequence => " *
           "$pname reward: $(p1_reward)\n"
    write_msg(loggers["eval"], msg)
end

function get_alphazero_vs_random_eval_fn(random_kwargs)

    function evaluate_alphazero_vs_random(loggers, nn, config, _)
        az_args = (init_cpu_mcts(nn, config),)
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
    end

    return evaluate_alphazero_vs_random
end

function get_nn_vs_random_eval_fn(random_kwargs)

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
    end

    return evaluate_nn_vs_random
end

function get_alphazero_vs_minimax_eval_fn(minimax_kwargs)

    function evaluate_alphazero_vs_minimax(loggers, nn, config, _)
        az_args = (init_cpu_mcts(nn, config),)
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
    end

    return evaluate_alphazero_vs_minimax
end

function get_nn_vs_minimax_eval_fn(minimax_kwargs)

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
    end

    return evaluate_nn_vs_minimax
end

function get_tictactoe_benchmark_fns(kwargs)
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

    mcts_wins_logger = LoggingUtilities.init_file_logger("mcts_correct.log"; overwrite=true)
    mcts_losses_logger = LoggingUtilities.init_file_logger("mcts_fails.log"; overwrite=true)

    nn_wins_logger = LoggingUtilities.init_file_logger("nn_correct.log"; overwrite=true)
    nn_losses_logger = LoggingUtilities.init_file_logger("nn_fails.log"; overwrite=true)

    function add_info_to_env_str(env_encoding, a, opt, idx)
        idx = lpad(idx, length(string(total_states)), "0")
        rows = split(env_encoding, "\n")
        fixed = "\n" *
                "             " * rows[1] * "\n" *
                "State $idx:  " * rows[2] * "\tAction chosen: $a. Optimal Actions: $opt\n" *
                "             " * rows[3] * "\n"
        return fixed
    end

    function benchmark_mcts_fn(loggers, nn, _, step)
        envs = DeviceArray(kwargs["eval-mcts-device"])(possible_envs)
        device_nn = (kwargs["eval-mcts-device"] == GPU()) ? Flux.gpu(nn) : Flux.cpu(nn)

        mcts = Policy(;
            device=kwargs["eval-mcts-device"],
            oracle=neural_network_env_oracle(; nn=device_nn),
            num_simulations=kwargs["eval-mcts-num-simulations"],
            value_scale=kwargs["eval-mcts-c_scale"],
            max_visit_init=kwargs["eval-mcts-c_visit"]
        )

        tree = MCTS.explore(mcts, envs)
        policy = MCTS.evaluation_policy(tree, mcts)

        log_msg(mcts_wins_logger, "Accuracy evaluation at step $step")
        log_msg(mcts_losses_logger, "Accuracy evaluation at step $step")

        false_positives = zeros(Int, 9)
        mcts_correct = 0
        for i in 1:total_states
            chosen_action = policy[i]
            env_encoding = string(possible_envs[i])
            info = add_info_to_env_str(env_encoding, chosen_action, optimal_actions[i], i)

            if chosen_action in optimal_actions[i]
                mcts_correct += 1
                write_msg(mcts_wins_logger, info)
            else
                false_positives[chosen_action] += 1
                write_msg(mcts_losses_logger, info)
            end
        end
        log_msg(mcts_losses_logger, "\nFalse positive actions: $false_positives\n")

        mcts_accuracy = round(mcts_correct / total_states, digits=4)
        with_logger(loggers["tb"]) do
            @info "eval" mcts_benchmark_accuracy=mcts_accuracy log_step_increment=0
        end
        log_msg(loggers["eval"], "MCTS Accuracy: $mcts_accuracy")
    end

    function benchmark_nn_fn(loggers, nn, _, step)
        cpu_nn = nn |> Flux.cpu
        _, logits = forward(cpu_nn, states, false)
        valid_actions = get_valid_actions(possible_envs)

        log_msg(nn_wins_logger, "Accuracy evaluation at step $step")
        log_msg(nn_losses_logger, "Accuracy evaluation at step $step")

        false_positives = zeros(Int, 9)
        nn_correct = 0
        for i in 1:total_states
            action_mask = typemin(Float32) .* .!valid_actions[:, i]
            masked_env_logits = logits[:, i] .+ action_mask

            chosen_action = argmax(masked_env_logits)
            env_encoding = string(possible_envs[i])
            info = add_info_to_env_str(env_encoding, chosen_action, optimal_actions[i], i)

            if chosen_action in optimal_actions[i]
                nn_correct += 1
                write_msg(nn_wins_logger, info)
            else
                false_positives[chosen_action] += 1
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
    end

    return benchmark_mcts_fn, benchmark_nn_fn
end

end
