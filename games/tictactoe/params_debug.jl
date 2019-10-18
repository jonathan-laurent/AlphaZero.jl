Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=100,
  depth_common=2)

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=30,
  update_threshold=(2 * 0.51 - 1),
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=0))

self_play = AlphaZero.SelfPlayParams(
  num_games=100,
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=20,
    dirichlet_noise_ϵ=0.15))

learning = AlphaZero.LearningParams()

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=8,
  num_game_stages=9)

validation = AlphaZero.RolloutsValidation(
  num_games = 100,
  baseline = AlphaZero.MctsParams(
    num_iters_per_turn=100,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
