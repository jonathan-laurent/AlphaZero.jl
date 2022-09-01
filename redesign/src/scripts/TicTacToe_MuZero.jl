import Random: MersenneTwister

using RLZero
using .Tests

function run_dummy_TTT_MZ_script()
    dummy_run_config = Config(
        BitwiseTicTacToeEnv,
        MuZeroTrainableEnvOracle,
        TrainSettings(;
            training_steps=1,
            training_envs=1,
            window_size=1,
            batch_size=1,
            nb_batches_per_training=1,
            num_unroll_steps=5,
            td_steps=9,
            discount=1,
            mcts_device=CPU(),
            explore=BatchedMcts.gumbel_explore,
        ),
        MersenneTwister(),
    )

    train(dummy_run_config)
end

function run_TTT_MZ_script()
    train_settings = TrainSettings(;
        training_steps=1000, # 1000
        training_envs=512, # 512
        window_size=1e6, # 1e6
        batch_size=512, # 512
        nb_batches_per_training=10, # 10
        num_unroll_steps=5,
        td_steps=9,
        discount=1,
        mcts_device=CPU(),
        explore=BatchedMcts.gumbel_explore,
    )

    config = Config(
        BitwiseTicTacToeEnv, MuZeroTrainableEnvOracle, train_settings, MersenneTwister()
    )
    train(config)
end


run_dummy_TTT_MZ_script()
@info "End of dummy run"
run_TTT_MZ_script()