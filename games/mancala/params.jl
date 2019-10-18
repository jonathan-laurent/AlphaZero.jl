Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=500,
  depth_common=4)

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=500, # 5000
  update_threshold=(2 * 0.55 - 1),
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=200,
    dirichlet_noise_ϵ=0.1))

self_play = AlphaZero.SelfPlayParams(
  num_games=20, # 200
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=200,
    dirichlet_noise_ϵ=0.15))

learning = AlphaZero.LearningParams(
  epochs_per_checkpoint=6,
  max_num_epochs=30,
  stop_loss_eps=-1.0) # practically equivalent to -Inf

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=10,
  num_game_stages=5)

validation = AlphaZero.RolloutsValidation(
  num_games = 500,
  baseline = AlphaZero.MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
