const RESNET = true

if RESNET
  Network = AlphaZero.ResNet{Game}
  netparams = AlphaZero.ResNetHP(
    num_filters=64,
    num_blocks=5,
    conv_kernel_size=(3,3),
    num_policy_head_filters=32,
    num_value_head_filters=32,
    batch_norm_momentum=0.5)
else
  Network = AlphaZero.SimpleNet{Game}
  netparams = AlphaZero.SimpleNetHP(
    width=500,
    depth_common=4)
end

self_play = AlphaZero.SelfPlayParams(
  num_games=4000,
  reset_mcts_every=4000,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=320,
    dirichlet_noise_ϵ=0.15))

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=1000,
  reset_mcts_every=1,
  update_threshold=(2*0.55 - 1),
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))

learning = AlphaZero.LearningParams(
  l2_regularization=1e-5,
  batch_size=128,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2, 5, 10, 20])

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=4,
  num_game_stages=9,
  mem_buffer_size=PLSchedule(
    [     0,      4],
    [20_000, 60_000]))

validation = AlphaZero.RolloutsValidation(
  num_games = 500,
  reset_mcts_every = 500,
  baseline = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
