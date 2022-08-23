"""
    Train

Main file containing the algorithm to `train` both AlphaZero & MuZero.
"""
module Train

import Base: @kwdef

using ..BatchedEnvs
using ..BatchedMcts # TODO: should be removed when the code is extended to be more generic
using ..Storage
using ..TrainableEnvOracleModule
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
    num_unroll_steps
    td_steps
    discount
    mcts_device
    explore
end

const ROOT = Int16(1)

function train(config)
    oracle = config.trainable_oracle()
    replay_buffer = ReplayBuffer(config.train_settings.window_size)

    for _ in 1:(config.train_settings.training_steps)
        games = play_games(config, oracle)
        save(replay_buffer, games)
        train_network(config.train_settings, oracle, replay_buffer)
    end
end

function play_games(config, trainable_oracle)
    envs = [config.game_env() for _ in 1:(config.train_settings.training_envs)]
    game_histories = [GameHistory() for _ in 1:(config.train_settings.training_envs)]
    mcts = BatchedMcts.Policy(;
        device=config.train_settings.mcts_device, oracle=get_EnvOracle(trainable_oracle)
    )

    while !all(terminated.(envs))
        previous_images = make_image.(envs)
        tree = config.train_settings.explore(mcts, envs, config.rng)

        values = tree.total_values[ROOT, :]
        policies = completed_qvalues(tree)
        actions = argmax.(policies)

        infos = act.(envs, actions)
        envs = first.(infos)
        rewards = map(info -> last(info).reward, infos)

        save.(game_histories, previous_images, actions, rewards, values, policies)
    end
    return game_histories
end

function train_network(train_settings, trainable_oracle, replay_buffer)
    batch = get_sample(replay_buffer, trainable_oracle, train_settings)
    update_weights(trainable_oracle, batch, train_settings)
    return nothing
end

end