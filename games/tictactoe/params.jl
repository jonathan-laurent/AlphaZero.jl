const RESNET = false

if RESNET
  Network = AlphaZero.ResNet{Game}
  netparams = AlphaZero.ResNetHP(
    num_filters=64,
    num_blocks=5,
    num_policy_head_filters=32,
    num_value_head_filters=32)
else
  Network = AlphaZero.SimpleNet{Game}
  netparams = AlphaZero.SimpleNetHP(
    width=500,
    depth_common=4)
end

self_play = AlphaZero.SelfPlayParams(
  num_games=6000,
  reset_mcts_every=3000,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=320,
    dirichlet_noise_ϵ=0.15))

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=500,
  reset_mcts_every=500,
  update_threshold=0.01,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))

learning = AlphaZero.LearningParams(
  l2_regularization=1e-5,
  batch_size=256,
  nonvalidity_penalty=1.,
  checkpoints=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=3,
  num_game_stages=9)

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
