module Train

using CUDA
using Flux
using Logging
using Random
using StaticArrays

using ...BatchedEnvs
using ...BatchedMcts
using ...BatchedMctsUtilities
using ...LoggingUtilities
using ...Network
using ...ReplayBuffers
using ...TrainUtilities
using ...Util.Devices

export selfplay!

const MCTS = BatchedMcts

"""
    step_save_reset!(config, envs, steps_counter, actions, ep_buff, rp_buff, γ=1f0)

Steps the provided actions in the environments, saves the transition data in the episode
buffer and resets the environments that have terminated. Data of terminated episodes is
transferred to the replay buffer.

# Arguments
- `config`: A `TrainConfig` object specifying the training parameters/hyperparameters.
- `envs`: The Array/CuArray containing the environments.
- `steps_counter`: The number-of-episode-steps counter for every environment.
- `actions`: The actions to take (step) in the environments.
- `ep_buff`: The episode buffer to save the transition data in.
- `rp_buff`: The replay buffer to transfer the data of terminated episodes to.
- `γ`: The discount factor used to compute state values of terminated episodes.
"""
function step_save_reset!(config, envs, steps_counter, actions, ep_buff, rp_buff, γ=1f0)
    # step
    out = act.(envs, actions)
    new_envs = first.(out)
    steps_counter .+= 1

    # save step data
    states = BatchedEnvs.vectorize_state.(envs)
    rewards = Float32.(map(x -> last(x).reward, out))
    switches = map(x -> last(x).switched, out)
    save!(ep_buff, states, actions, rewards, switches)

    # get terminated envs
    dones = DeviceArray(get_device(envs))(BatchedEnvs.terminated.(new_envs))

    # transfer data of terminated episodes to replay buffer
    dones_cpu = Array(dones)
    map(1:config.num_envs) do env_id
        @inbounds if dones_cpu[env_id]
            # compute rewards-to-go (value functions)
            compute_value_functions!(ep_buff, env_id, γ)

            # transfer sample to replay buffer and reset episode buffer
            add!(rp_buff, ep_buff, env_id)
            empty_env_in_buffer!(ep_buff, env_id)
        end
    end

    # reset terminated envs
    Devices.foreach(1:config.num_envs, get_device(envs)) do env_id
        @inbounds if dones[env_id]
            new_envs[env_id] = BatchedEnvs.reset(envs[env_id])
            steps_counter[env_id] = 0
        end
    end

    return new_envs
end

"""
    alphazero_loss(nn, states, actions, state_values)

Computes the loss of the AlphaZero network: L(θ) = L_v(θ) + L_p(θ) where
    - L_v(θ) = (v - v_target)²   (mean squared error)
    - L_p(θ) = -πᵀ × log(p)      (cross-entropy)
are the value and policy losses respectively.

# Arguments
- `nn`: The neural network.
- `states`: Environment states. Size: (state_size..., batch_size)
- `actions`: One-hot encodings of actions taken. Size: (num_actions, batch_size)
- `state_values`: The value function targets. Size: (1, batch_size)
"""
function alphazero_loss(nn, states, actions, state_values)
    pred_v, pred_logits = forward(nn, states, false)
    value_loss = Flux.mse(pred_v, state_values)
    policy_loss = Flux.logitcrossentropy(pred_logits, actions)
    loss = value_loss + policy_loss
    return value_loss, policy_loss, loss
end

"""
    train!(nn, replay_buffer, opt, device, config, loggers, rng)

Trains and returns the provided neural network on the provided replay buffer.

# Arguments
- `nn`: The neural network to train.
- `replay_buffer`: The replay buffer containing the training data.
- `opt`: The optimizer to use for training.
- `device`: The device on which to train the network.
- `config`: The configuration object containing the training parameters.
- `loggers`: The dictionary of loggers to use for logging the training losses.
- `rng`: The random number generator used to shuffle training data.
"""
function train!(nn, replay_buffer, opt, device, config, loggers, rng)
    # get the data in a meaningul format
    states, actions, state_values = to_array(replay_buffer)

    # create a dataloader
    data = (states, actions, state_values)
    dataloader = Flux.DataLoader(data, batchsize=config.batch_size, shuffle=true, rng=rng)

    # train the model for the specified number of epochs
    for _ in 1:config.train_epochs
        for batch in dataloader
            # if the batch has only 1 element, it will trigger NaN in BatchNorm -> skip
            (size(batch[1])[end] == 1) && continue

            # convert to train device
            batch = map(DeviceArray(device), batch)

            # hack: predefine variables so they get assigned correctly inside the do block
            val_loss, pol_loss = 0f0, 0f0
            loss, grads = Flux.withgradient(nn) do model
                val_loss, pol_loss, loss = alphazero_loss(model, batch...)
                return loss
            end
            "tb" in keys(loggers) && log_losses(loggers["tb"], val_loss, pol_loss, loss)
            _, nn = Flux.update!(opt, nn, grads[1])
        end
    end

    return nn
end

"""
    selfplay!(config, device, nn)

Runs the AlphaZero selfplay training loop. Returns the trained neural network and a
`TrainExecutionTimes` object containing the execution times of the different steps of the
training loop.

# Arguments
- `config`: A `TrainConfig` object specifying the training parameters/hyperparameters.
- `device`: The device on which to run MCTS and NN training.
- `nn`: The neural network to train.
- `print_progress`: Whether to print progress messages in stdout.

# Returns
- `nn`: The trained neural network.
- `times`: A `TrainExecutionTimes` object.
"""
function selfplay!(config, device, nn, print_progress=true)
    state_size = BatchedEnvs.state_size(config.EnvCls)
    num_actions = BatchedEnvs.num_actions(config.EnvCls)

    adam = Flux.Optimiser(
        Flux.ClipValue(config.gradient_clip),
        Flux.WeightDecay(config.weight_decay),
        Flux.Adam(config.adam_lr)
    )
    opt = Flux.setup(adam, nn)

    envs, steps_counter = init_envs(config, config.num_envs, device)

    batch_steps = config.num_steps ÷ config.num_envs
    batch_train_freq = config.train_freq ÷ config.num_envs
    batch_eval_freq = config.eval_freq ÷ config.num_envs

    ep_buff = EpisodeBuffer(config.num_envs, state_size)
    rp_buff = ReplayBuffer(config.replay_buffer_size, state_size, num_actions)

    loggers = init_loggers(config; overwrite_logfiles=true, overwrite_tb_logdir=true)

    mcts_rng = Random.MersenneTwister(3409)
    train_rng = Random.MersenneTwister(3409)

    (config.nn_save_dir != "") && save_nn(nn, config.nn_save_dir, 0, batch_steps)

    times = TrainExecutionTimes(batch_steps)
    for step in 1:batch_steps
        print_progress && println("Step: $step.")

        # instiantiate mcts config object
        mcts_config = init_mcts_config(device, nn, config)

        # run mcts
        if config.use_gumbel_mcts
            t = @elapsed tree, gumbel = MCTS.gumbel_explore(mcts_config, envs, mcts_rng)
            times.explore_times[step] = t

            t = @elapsed actions = MCTS.gumbel_policy(tree, mcts_config, gumbel)
            times.selection_times[step] = t
        else
            t = @elapsed tree = MCTS.alphazero_explore(mcts_config, envs, mcts_rng)
            times.explore_times[step] = t

            t = @elapsed begin
                actions = MCTS.alphazero_policy(tree, mcts_config, steps_counter, mcts_rng)
            end
            times.selection_times[step] = t
        end

        # step, save data from terminated episodes and reset them
        t = @elapsed begin
            envs .= step_save_reset!(config, envs, steps_counter, actions, ep_buff, rp_buff)
        end
        times.step_save_reset_times[step] = t

        # train the network if it's time to do so
        if step % batch_train_freq == 0 && length(rp_buff) >= config.min_train_samples
            print_progress && println("Training with $(length(rp_buff)) samples.")
            t = @elapsed begin
                set_train_mode!(nn)
                nn = train!(nn, rp_buff, opt, device, config, loggers, train_rng)
                set_test_mode!(nn)
            end
            times.train_times[step] = t
            (config.nn_save_dir != "") && save_nn(nn, config.nn_save_dir, step, batch_steps)
        end

        # evaluate the network if it's time to do so
        if batch_eval_freq > 0 && step % batch_eval_freq == 0 && length(config.eval_fns) > 0
            "eval" in keys(loggers) && log_msg(loggers["eval"], "Evaluating at step: $step")
            t = @elapsed begin
                for eval_fn in config.eval_fns
                    eval_fn(loggers, nn, config, step)
                end
            end
            times.eval_times[step] = t
            "eval" in keys(loggers) && write_msg(loggers["eval"], "\n")
        end
    end

    return nn, times
end



end
