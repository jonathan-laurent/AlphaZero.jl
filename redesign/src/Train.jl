module Train

using Base: @kwdef
using BenchmarkTools
using CUDA
using Flux
using JLD2
using Logging
using Random
using StaticArrays

using ...BatchedEnvs
using ...BatchedMcts
using ...BatchedMctsUtilities: Policy
using ...EnvOracles
using ...Evaluation
using ...LoggingUtilities
using ...Network
using ...ReplayBuffers
using ...Util.Devices
using ...Util.Devices.KernelFuns: argmax

export TrainConfig, selfplay!

const MCTS = BatchedMcts


@kwdef struct TrainConfig
    EnvCls::Type
    env_kwargs::Dict
    num_envs::Int
    num_simulations::Int
    replay_buffer_size::Int
    mcts_value_scale::Float32
    train_freq::Int
    min_train_samples::Int
    adam_lr::Float32
    weight_decay::Float32 = 0f0
    gradient_clip::Float32
    batch_size::Int
    train_epochs::Int
    train_logfile::String
    eval_logfile::String
    tb_logdir::String
    eval_fns::Vector{Function} = []
    eval_freq::Int
    num_steps::Int
end


function init_envs(config, num_envs, device)
    envs = [config.EnvCls(; config.env_kwargs...) for _ in 1:num_envs]
    return DeviceArray(device)(envs)
end


function init_mcts(device::Device, nn::Net, config::TrainConfig) where Net <: FluxNetwork
    oracle = neural_network_env_oracle(; nn)
    return Policy(;
        device,
        oracle,
        num_simulations=config.num_simulations,
        value_scale=config.mcts_value_scale
    )
end


function step_save_and_reset!(config, envs, actions, episode_buffer, replay_buffer, γ=1f0)
    out = act.(envs, actions)
    new_envs = first.(out)

    states = BatchedEnvs.vectorize_state.(envs)
    rewards = Float32.(map(x -> last(x).reward, out))
    switches = map(x -> last(x).switched, out)
    save!(episode_buffer, states, actions, rewards, switches)

    map(1:config.num_envs) do env_id
        @inbounds done = BatchedEnvs.terminated(new_envs[env_id])
        if done
            # propagate rewards (for sparse-reward games)
            # propagate_reward!(episode_buffer, env_id)  # ToDo: fix for all settings
            compute_value_functions!(episode_buffer, env_id, γ)

            # reset
            add!(replay_buffer, episode_buffer, env_id)
            empty_env_in_buffer!(episode_buffer, env_id)
            @inbounds new_envs[env_id] = config.EnvCls(; config.env_kwargs...)
        end
    end

    return new_envs
end


function alphazero_loss(nn, states, actions, rewards)
    pred_v, pred_logits = forward(nn, states, false)
    value_loss = Flux.mse(pred_v, rewards)
    policy_loss = Flux.logitcrossentropy(pred_logits, actions)
    loss = value_loss + policy_loss
    return value_loss, policy_loss, loss
end


function train!(nn, replay_buffer, opt, device, config, loggers)
    # get the data in a meaningul format
    states, actions, rewards = to_array(replay_buffer, device)

    # create a dataloader
    data = (states, actions, rewards)
    rng = Random.MersenneTwister(0)
    dataloader = Flux.DataLoader(data, batchsize=config.batch_size, shuffle=true, rng=rng)

    # train the model for the specified number of epochs
    for _ in 1:config.train_epochs
        for batch in dataloader
            # if batch contains only 1 element, it will trigger NaN in BatchNorm -> skip
            current_batch_size = size(batch[1])[end]
            (current_batch_size == 1) && continue

            # hack: predefine variables so they get assigned correctly inside the do block
            value_loss, policy_loss = 0f0, 0f0
            loss, grads = Flux.withgradient(nn) do model
                value_loss, policy_loss, loss = alphazero_loss(model, batch...)
                return loss
            end
            log_losses(loggers["tb"], value_loss, policy_loss, loss)
            _, nn = Flux.update!(opt, nn, grads[1])
        end
    end

    return nn
end


function save_nn(nn; step, total_steps)
    model_state = Flux.state(Flux.cpu(nn))
    step_str = lpad(step, length(string(total_steps)), "0")
    jldsave("checkpoints/model_$step_str.jld2"; model_state)
end


function selfplay!(config, device, nn)

    state_size = BatchedEnvs.state_size(config.EnvCls)
    num_actions = BatchedEnvs.num_actions(config.EnvCls)

    adam = Flux.Optimiser(
        Flux.ClipValue(config.gradient_clip),
        Flux.WeightDecay(config.weight_decay),
        Flux.Adam(config.adam_lr)
    )
    opt = Flux.setup(adam, nn)

    envs = init_envs(config, config.num_envs, device)

    batch_env_steps = config.num_steps ÷ config.num_envs
    size_per_env = config.replay_buffer_size ÷ config.num_envs
    batch_train_freq = config.train_freq ÷ config.num_envs
    batch_eval_freq = config.eval_freq ÷ config.num_envs

    episode_buffer = EpisodeBuffer(config.num_envs)
    replay_buffer = ReplayBuffer(config.num_envs, size_per_env, state_size, num_actions)

    loggers = init_loggers(config; overwrite_logfiles=true, overwrite_tb_logdir=true)

    gumbel_mcts_rng = Random.MersenneTwister(3409)

    save_nn(nn; step=0, total_steps=batch_env_steps)

    for step in 1:batch_env_steps
        println("Step: $step")

        # run mcts simulations to get actions
        mcts = init_mcts(device, nn, config)
        tree, gumbel = MCTS.gumbel_explore(mcts, envs, gumbel_mcts_rng)
        actions = MCTS.gumbel_policy(tree, mcts, gumbel)

        # step, save data from terminated episodes and reset them
        envs .= step_save_and_reset!(config, envs, actions, episode_buffer, replay_buffer)

        # train the network if it's time to do so
        if step % batch_train_freq == 0 && length(replay_buffer) >= config.min_train_samples
            println("Training at step: $step, len(replay_buffer): $(length(replay_buffer))")
            set_train_mode!(nn)
            nn = train!(nn, replay_buffer, opt, device, config, loggers)
            set_test_mode!(nn)
            save_nn(nn; step, total_steps=batch_env_steps)
        end

        # evaluate the network if it's time to do so
        if step % batch_eval_freq == 0 && length(config.eval_fns) > 0
            "eval" in keys(loggers) && log_msg(loggers["eval"], "Evaluating at step: $step")
            for eval_fn in config.eval_fns
                eval_fn(loggers, nn, config, step)
            end
            write_msg(loggers["eval"], "\n")
        end
    end
end



end
