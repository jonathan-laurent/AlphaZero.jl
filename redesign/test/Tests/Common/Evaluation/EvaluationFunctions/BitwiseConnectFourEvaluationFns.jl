module BitwiseConnectFourEvalFns

using Flux
using Logging
using Random

using ..BitwiseConnectFour
using ..BitwiseConnectFourHeuristic
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
export get_connect_four_benchmark_fns

const MCTS = BatchedMcts


function act_minimax(single_env_vec, kwargs)
    env = Array(single_env_vec)[1]
    rng = kwargs["stochastic_minimax_rng"]
    depth = kwargs["stochastic_minimax_depth"]
    eval_fn = connect_four_eval_fn
    is_maximizing = env.curplayer == BitwiseConnectFour.CROSS
    return stochastic_minimax(env, rng; depth, eval_fn, is_maximizing, amplify_rewards=true)
end

function write_episode_action_sequence(loggers, episode, action_sequence, p1_reward, pname)
    episode_str = lpad(episode, 2, "0")
    msg = "\tEpisode $episode_str:\tAction sequence: $action_sequence => " *
           "$pname reward: $(p1_reward)\n"
    write_msg(loggers["eval"], msg)
end

function get_alphazero_vs_random_eval_fn(random_kwargs)

    function evaluate_alphazero_vs_random(loggers, nn, _, _)
        az_args = (init_mcts_config(CPU(), Flux.cpu(nn), random_kwargs["config"]),)
        az_wins, az_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of AlphaZero vs random policy")
        for iter in 1:random_kwargs["num_bilateral_rounds"]
            env = BitwiseConnectFourEnv()

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
            env = BitwiseConnectFourEnv()

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

    function evaluate_alphazero_vs_minimax(loggers, nn, _, _)
        az_args = (init_mcts_config(CPU(), Flux.cpu(nn), minimax_kwargs["config"]),)
        az_wins, az_losses, draws = 0, 0, 0
        log_msg(loggers["eval"], "Starting evaluation of AlphaZero vs minimax policy")
        for iter in 1:minimax_kwargs["num_bilateral_rounds"]
            env = BitwiseConnectFourEnv()

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
            env = BitwiseConnectFourEnv()

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

function _create_pos_env(action_list)
    env = BitwiseConnectFourEnv()
    for action in action_list
        env, _ = BatchedEnvs.act(env, action)
    end
    return env
end

"""
    _sample_states(rng, num_states)

Samples randomly `num_states` states, computes the optimal policy using Pascal Pons' solver:
https://github.com/jonathan-laurent/AlphaZero.jl/blob/7b5cf057ce699b81ce948d22f5bb2ebd60fe6c56/games/connect-four/solver/README.md
and returns the states, the optimal actions and the optimal policy.
"""
function _sample_states(rng, num_states)
    na = BatchedEnvs.num_actions(BitwiseConnectFourEnv)
    maxlen = BitwiseConnectFour.NUM_ROWS * BitwiseConnectFour.NUM_COLUMNS

    action_sequences = rand(rng, 1:na, (num_states, maxlen))

    final_envs = Vector{BitwiseConnectFourEnv}()
    final_states = []
    final_optimal_actions = []

    cmd = pipeline(Cmd(`./c4solver`, dir="cf_stuff/connect4"), stderr=devnull)
    process = open(cmd, "r+")

    for i in 1:num_states
        action_sequence = action_sequences[i, :]
        episode_actions = []

        env = BitwiseConnectFourEnv()
        for action in action_sequence
            !BatchedEnvs.valid_action(env, action) && continue
            env, _ = BatchedEnvs.act(env, action)
            BatchedEnvs.terminated(env) && break
            push!(episode_actions, action)
        end
        random_len = rand(rng, 4:length(episode_actions))

        random_action_subsequence = episode_actions[1:random_len]
        generated_env = _create_pos_env(random_action_subsequence)

        random_action_subsequence_str = join(random_action_subsequence)
        min_value = typemax(Float32)
        values = fill(min_value, na)
        num_valid_actions = 0
        for a in 1:na
            BatchedEnvs.valid_action(generated_env, a) ? (num_valid_actions += 1) : continue
            _, info = BatchedEnvs.act(generated_env, a)
            if info.reward != 0
                score = Minimax.amplify(-info.reward)
                values[a] = score
                min_value = min(min_value, score)
                continue
            end
            input = random_action_subsequence_str * string(a)
            println(process, input)
            line = readline(process)
            score = parse(Float32, split(line)[end])
            values[a] = score
            min_value = min(min_value, score)
        end
        (num_valid_actions == 1) && continue
        optimal_actions = [action for action in 1:na if values[action] == min_value]

        push!(final_envs, generated_env)
        push!(final_states, Array(BatchedEnvs.vectorize_state(generated_env)))
        push!(final_optimal_actions, optimal_actions)
    end

    close(process)

    states_cat = hcat(final_states...)
    return final_envs, states_cat, final_optimal_actions
end

function get_connect_four_benchmark_fns(kwargs)

    println("Initializing benchmark data...")
    all_envs, states, optimal_actions = _sample_states(kwargs["rng"], kwargs["num_states"])
    total_states = length(all_envs)
    println("Initialized $total_states environments.")

    mcts_losses_logger = LoggingUtilities.init_file_logger("mcts_fails.log"; overwrite=true)
    nn_losses_logger = LoggingUtilities.init_file_logger("nn_fails.log"; overwrite=true)

    function add_info_to_env_str(env, a, opt, idx)
        env_encoding = string(env)
        idx = lpad(idx, length(string(total_states)), "0")
        fixed = "State $idx:\n$env_encoding\n" *
                "Action chosen: $a. Optimal Actions: $opt\n"
        return fixed
    end

    function benchmark_mcts_fn(loggers, nn, _, step)
        envs = DeviceArray(kwargs["device"])(all_envs)
        device_nn = (kwargs["device"] == GPU()) ? Flux.gpu(nn) : Flux.cpu(nn)
        mcts_config = init_mcts_config(kwargs["device"], device_nn, kwargs["config"])

        tree = MCTS.explore(mcts_config, envs)
        policy = Array(MCTS.evaluation_policy(tree, mcts_config))
        children_visits = Array(MCTS.get_root_children_visits(tree, mcts_config))

        log_msg(mcts_losses_logger, "Accuracy evaluation at step $step")

        true_positives, false_positives = zeros(Int, 7), zeros(Int, 7)
        mcts_correct = 0
        for i in 1:total_states
            chosen_action = policy[i]

            if chosen_action in optimal_actions[i]
                mcts_correct += 1
                true_positives[chosen_action] += 1
            else
                false_positives[chosen_action] += 1
                s = add_info_to_env_str(all_envs[i], chosen_action, optimal_actions[i], i)
                write_msg(mcts_losses_logger, s)
                write_msg(mcts_losses_logger, "Visit counts: $(children_visits[:, i])")
                write_msg(mcts_losses_logger, "\n\n\n")
            end
        end
        log_msg(mcts_losses_logger, "\nTrue positive actions: $true_positives\n")
        log_msg(mcts_losses_logger, "False positive actions: $false_positives\n")

        mcts_accuracy = round(mcts_correct / total_states, digits=4)
        with_logger(loggers["tb"]) do
            @info "eval" mcts_benchmark_accuracy=mcts_accuracy log_step_increment=0
        end
        log_msg(loggers["eval"], "MCTS Accuracy: $mcts_accuracy")
    end

    function benchmark_nn_fn(loggers, nn, _, step)
        cpu_nn = nn |> Flux.cpu
        _, logits = forward(cpu_nn, states, false)
        valid_actions = get_valid_actions(all_envs)

        log_msg(nn_losses_logger, "Accuracy evaluation at step $step")

        true_positives, false_positives = zeros(Int, 7), zeros(Int, 7)
        nn_correct = 0
        for i in 1:total_states
            action_mask = typemin(Float32) .* .!valid_actions[:, i]
            masked_env_logits = logits[:, i] .+ action_mask

            chosen_action = argmax(masked_env_logits)
            if chosen_action in optimal_actions[i]
                nn_correct += 1
                true_positives[chosen_action] += 1
            else
                false_positives[chosen_action] += 1
                s = add_info_to_env_str(all_envs[i], chosen_action, optimal_actions[i], i)
                write_msg(nn_losses_logger, s)
                write_msg(nn_losses_logger, "Masked Logits: $masked_env_logits\n\n\n")
            end
        end
        log_msg(nn_losses_logger, "\nTrue positive actions: $true_positives\n")
        log_msg(nn_losses_logger, "False positive actions: $false_positives\n")

        nn_accuracy = round(nn_correct / total_states, digits=4)
        with_logger(loggers["tb"]) do
            @info "eval" nn_benchmark_accuracy=nn_accuracy log_step_increment=0
        end
        log_msg(loggers["eval"], "Raw NN Accuracy: $nn_accuracy")
    end

    return benchmark_mcts_fn, benchmark_nn_fn
end


end
