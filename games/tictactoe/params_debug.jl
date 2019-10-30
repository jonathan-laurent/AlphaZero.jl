# Does not work very well as MCTS is reset at every step...


Network = AlphaZero.ResNet{Game}

netparams = AlphaZero.ResNetHP(
  num_filters=64,
  num_blocks=5,
  num_policy_head_filters = 32,
  num_value_head_filters = 32)

#=
Network = AlphaZero.SimpleNet{Game}
netparams = AlphaZero.SimpleNetHP(
  width=600,
  depth_common=4)
=#

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=400,
  reset_mcts_every=50,
  update_threshold=(2 * 0.51 - 1),
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=0))

self_play = AlphaZero.SelfPlayParams(
  num_games=100,
  reset_mcts_every=50,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    use_gpu=true,
    num_iters_per_turn=20,
    dirichlet_noise_ϵ=0.15))

learning = AlphaZero.LearningParams(
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[10, 20])

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=8,
  num_game_stages=9)

validation = AlphaZero.RolloutsValidation(
  num_games = 100,
  reset_mcts_every=100,
  baseline = AlphaZero.MctsParams(
    num_iters_per_turn=100,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
