"""
# Evaluation Module

## Overview

This module provides a suite of functionalities for quantitative assessment of
neural network and MCTS agents. It allows the agents to be evaluated against custom-defined
agents or functions, including, but not limited to, random agents and minimax agents with
heuristics. This module facilitates the ongoing monitoring of agent performance metrics
serving as a more reliable gauge of efficacy compared to loss functions or reward signals
alone.

These evaluation methods are designed to be invoked at intervals specified by
`batch_eval_freq` during the training loop. The invocation occurs in the `selfplay!()`
function of the `Train.jl` module as follows:

```julia
if batch_eval_freq > 0 && step % batch_eval_freq == 0 && length(config.eval_fns) > 0
    for eval_fn in config.eval_fns
        eval_fn(loggers, nn, config, step)
    end
end

## Function Signature

The user-defined evaluation functions (eval_fn) should adhere to the following function
signature:

```julia
function eval_fn(
    loggers::Dict,
    nn::Net,
    config::TrainConfig,
    step::Int
) where Net <: FluxNetwork
    # Function implementation
end
```

## Arguments

- `loggers`: A Dict with loggers, as returned by `init_loggers()` in LoggingUtilities.jl.
- `nn`: The neural network to evaluate or use for evaluating MCTS.
- `config`: The `TrainConfig` object that was provided to `selfplay!()`.
- `step`: The current (batch) step in the training loop.

## Implentation Notes

1. Parameterization via Closures: To facilitate code reusability, it is advised to
   parameterize the evaluation functions using closures, as illustrated below:

    ```julia
    function get_eval_fn(my_args)
        function eval_fn(loggers, nn, config, step)
            # Utilize my_args in evaluation
        end
        return eval_fn
    end
   ```

2. Custom Logging: For specialized logging needs, one may consider creating custom loggers,
   aside from the built-in ones instantiated via init_loggers(). For efficient logging
   mechanisms that integrate with platforms such as TensorBoard, refer to
   LoggingUtilities.jl.
"""
module Evaluation

using Flux
using Logging
using Random

using ...BatchedEnvs
using ...BatchedMcts
using ...BatchedMctsUtilities
using ...Minimax
using ...Network
using ...TrainUtilities
using ...Util.Devices

const MCTS = BatchedMcts


export mcts_move, nn_move, random_move, minimax_move
export play_round
export get_alphazero_vs_custom_fn_eval_fn, get_nn_vs_custom_fn_eval_fn


function mcts_move(single_env_vec, mcts_config)
    device_single_env_vec = DeviceArray(mcts_config.device)(single_env_vec)
    tree = MCTS.explore(mcts_config, device_single_env_vec)
    return Array(MCTS.evaluation_policy(tree, mcts_config))[1]
end

function nn_move(single_env_vec, cpu_nn)
    state = BatchedEnvs.vectorize_state(single_env_vec[1])
    state = Flux.unsqueeze(state, length(size(state)) + 1)
    _, p = forward(cpu_nn, state)
    invalid_actions = .!get_valid_actions(single_env_vec)[:, 1]
    p[invalid_actions, 1] .= -Inf
    return argmax(p[:, 1])
end

function random_move(single_env_vec, kwargs)
    valid_actions_bool = Array(get_valid_actions(single_env_vec))[:, 1]
    valid_actions = findall(valid_actions_bool)
    return rand(kwargs["rng"], valid_actions)
end

function minimax_move(single_env_vec, kwargs)
    return minimax(single_env_vec[1]; kwargs...)
end

function play_round(env, az_or_nn_fn, az_or_nn_args, custom_fn, custom_fn_kwargs, az_starts)
    single_env_vec = [env]
    current_player_is_az = az_starts

    total_reward, done, info = 0.0, false, nothing
    action_sequence = []
    while !done
        if current_player_is_az
            action = az_or_nn_fn(single_env_vec, az_or_nn_args...)
        else
            action = custom_fn(single_env_vec, custom_fn_kwargs)
        end
        push!(action_sequence, action)

        new_env, info = BatchedEnvs.act(single_env_vec[1], action)
        total_reward += info.reward
        single_env_vec[1] = new_env
        done = BatchedEnvs.terminated(single_env_vec[1])
        (info.switched && !done) && (current_player_is_az = !current_player_is_az)
    end

    az_perspective_reward = (current_player_is_az) ? (total_reward) : (-total_reward)
    return az_perspective_reward, action_sequence
end

function get_alphazero_vs_custom_fn_eval_fn(custom_fn_kwargs)

    function evaluate_alphazero_vs_custom_fn(loggers, nn, config, _)
        custom_fn = custom_fn_kwargs["custom_fn"]
        az_args = (init_mcts(custom_fn_kwargs["device"], nn, custom_fn_kwargs["config"]),)
        az_wins, az_losses, draws = 0, 0, 0

        # bilateral means: 1 game where az plays first and 1 game where az plays second
        num_bilateral_games = 10
        for iter in 1:num_bilateral_games
            env = config.EnvCls(; config.env_kwargs...)
            custom_fn_kwargs["iteration"] = iter

            # az plays first
            r, _ = play_round(env, mcts_move, az_args, custom_fn, custom_fn_kwargs, true)
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_wins += 1) : (az_losses += 1))

            # az plays second
            r, _ = play_round(env, mcts_move, az_args, custom_fn, custom_fn_kwargs, false)
            (r == 0) ? (draws += 1) : ((r > 0) ? (az_losses += 1) : (az_wins += 1))
        end

        win_rate = az_wins / (2 * num_bilateral_games)
        loss_rate = az_losses / (2 * num_bilateral_games)
        draw_rate = draws / (2 * num_bilateral_games)
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_az_vs_custom_fn=win_rate log_step_increment=0
            @info "eval" loss_rate_az_vs_custom_fn=loss_rate log_step_increment=0
            @info "eval" draw_rate_az_vs_custom_fn=draw_rate log_step_increment=0
        end
    end

    return evaluate_alphazero_vs_custom_fn
end

function get_nn_vs_custom_fn_eval_fn(custom_fn_kwargs)

    function evaluate_nn_vs_custom_fn(loggers, nn, config, _)
        cpu_nn = nn |> Flux.cpu
        custom_fn = custom_fn_kwargs["custom_fn"]
        nn_wins, nn_losses, draws = 0, 0, 0
        num_bilateral_games = 10
        for iter in 1:num_bilateral_games
            env = config.EnvCls(; config.env_kwargs...)
            custom_fn_kwargs["iteration"] = iter

            # nn plays first
            r, _ = play_round(env, nn_move, (cpu_nn,), custom_fn, custom_fn_kwargs, true)
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_wins += 1) : (nn_losses += 1))

            # nn plays second
            r, _ = play_round(env, nn_move, (cpu_nn,), custom_fn, custom_fn_kwargs, false)
            (r == 0) ? (draws += 1) : ((r > 0) ? (nn_losses += 1) : (nn_wins += 1))
        end

        win_rate = nn_wins / (2 * num_bilateral_games)
        loss_rate = nn_losses / (2 * num_bilateral_games)
        draw_rate = draws / (2 * num_bilateral_games)
        with_logger(loggers["tb"]) do
            @info "eval" win_rate_nn_vs_custom_fn=win_rate log_step_increment=0
            @info "eval" loss_rate_nn_vs_custom_fn=loss_rate log_step_increment=0
            @info "eval" draw_rate_nn_vs_custom_fn=draw_rate log_step_increment=0
        end
    end

    return evaluate_nn_vs_custom_fn
end

end
