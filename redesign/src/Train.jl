"""
    Train

Main file containing the algorithm to `train` both AlphaZero & MuZero.
"""
module Train

import Base: @kwdef

using ..BatchedEnvs
using ..Storage
using ..TrainableEnvOracleModule

export Config, TrainSettings, train

"""
    Config
    
All the required informations to train (& infere) on AlphaZero/ MuZero.
"""
@kwdef struct Config
    game_env
    trainable_oracle
    train_settings
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
    mcts_device
    explore
end

function train(config)
    oracle = config.trainable_oracle()
    replay_buffer = ReplayBuffer(config.train_settings.window_size)

    for _ in 1:(config.train_settings.training_steps)
        games = play_games(config, oracle)
        save(replay_buffer, games)
        train(config.train_settings, oracle, replay_buffer)
    end
end

function play_games(config, trainable_oracle)
    envs = [config.game_env() for _ in 1:(config.train_settings.training_envs)]
    game_histories = [
        GameHistory(config.game_env) for _ in 1:(config.train_settings.training_envs)
    ]
    mcts = Policy(;
        device=config.train_settings.mcts_device, oracle=get_EnvOracle(trainable_oracle)
    )

    while !all(terminated.(envs))
        previous_images = make_image.(envs)
        tree = config.train_settings.explore(mcts, envs, config.rng)

        infos = act.(envs, actions)
        envs = first.(infos)
        rewards = broadcast(info -> last(info).reward, infos)

        values = tree.total_values[ROOT, :]
        policies = completed_qvalues(tree)
        actions = argmax.(policies)

        save.(game_histories, previous_images, actions, rewards, values, policies)
    end
    return game_histories
end

function train(train_settings, trainable_oracle, replay_buffer)
    batch = get_sample(replay_buffer, trainable_oracle, train_settings)
    update_weights(trainable_oracle, batch, train_settings)
    return nothing
end

end