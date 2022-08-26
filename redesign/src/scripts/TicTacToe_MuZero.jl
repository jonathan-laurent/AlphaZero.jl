import Random: MersenneTwister

using RLZero
using .Tests

train_settings = TrainSettings(; # Fake hyperparameters
    training_steps=10,
    training_envs=10,
    window_size=100,
    batch_size=10,
    nb_batches_per_training=1,
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