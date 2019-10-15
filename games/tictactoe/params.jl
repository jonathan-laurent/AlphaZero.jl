Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=500,
  depth_common=4)

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=3000,
  update_threshold=(2 * 0.51 - 1),
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=0))

self_play = AlphaZero.SelfPlayParams(
  num_games=2000,
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=200,
    dirichlet_noise_ϵ=0.15))

learning = AlphaZero.LearningParams()

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=4,
  num_game_stages=9)

validation = AlphaZero.RolloutsValidation(
  num_games = 1000,
  baseline = AlphaZero.MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
