"""
    Train

Main file containing the algorithm to `train` both AlphaZero & MuZero.
"""
module Train

import Base: @kwdef

using ..BatchedEnvs
using ..BatchedMcts # TODO: should be removed when the code is extended to be more generic
using ..Storage
using ..TrainableEnvOracles
using ..MuZero

export Config, TrainSettings, train

"""
    Config
    
All the required informations to train (& infere) on AlphaZero/ MuZero.
"""
@kwdef struct Config
    game_env
    trainable_oracle
    train_settings
    rng
end

"""
    TrainSettings
    
Training-related settings.
"""
@kwdef struct TrainSettings
    training_steps
    training_envs
    window_size
    batch_size
    nb_batches_per_training
    num_unroll_steps
    td_steps
    discount
    mcts_device
    explore
end

const ROOT = Int16(1)

function train(config)
    @info "Start agent training."
    oracle = config.trainable_oracle()
    replay_buffer = ReplayBuffer(config.train_settings.window_size)

    for iter in 1:(config.train_settings.training_steps)
        @info "Start training iteration no. $iter."

        (; time) = @timed games = play_games(config, oracle)
        save(replay_buffer, games)
        @info "  Self-Play ended in $(time)s."

        (; time) = @timed train_iteration(config.train_settings, oracle, replay_buffer)
        @info "  Network training ended in $(time)s."
    end
    @info "Finished agent training."
end

function play_games(config, trainable_oracle)
    envs = [config.game_env() for _ in 1:(config.train_settings.training_envs)]
    final_game_histories = [GameHistory() for _ in 1:(config.train_settings.training_envs)]
    mcts = BatchedMcts.Policy(;
        device=config.train_settings.mcts_device, oracle=get_env_oracle(trainable_oracle)
    )

    game_histories = final_game_histories
    while !isempty(envs)
        states = vectorize_state.(envs)
        tree = config.train_settings.explore(mcts, envs, config.rng)

        values = tree.total_values[ROOT, :]
        policies = completed_qvalues(tree)
        actions = argmax.(policies)

        infos = act.(envs, actions)
        envs = first.(infos)
        rewards = map(info -> last(info).reward, infos)

        save.(game_histories, states, actions, rewards, values, policies)

        envs = [env for env in envs if !terminated(env)]
        game_histories = [
            history for (env, history) in zip(envs, game_histories) if !terminated(env)
        ]
    end
    return final_game_histories
end

function train_iteration(train_settings, trainable_oracle, replay_buffer)
    batches = [
        get_sample(replay_buffer, trainable_oracle, train_settings) for
        _ in 1:(train_settings.nb_batches_per_training)
    ]
    update_weights(trainable_oracle, batches, train_settings)
    return nothing
end

end