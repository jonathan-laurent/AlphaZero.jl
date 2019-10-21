Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=600,
  depth_common=3,
  depth_vbranch=2,
  depth_pbranch=2)

mcts = AlphaZero.MctsParams(
  num_workers=64,
  num_iters_per_turn=640,
  dirichlet_noise_epsilon=0)

self_play = AlphaZero.SelfPlayParams(
  num_games=200,
  mcts = mcts)

# Exploration is induced by MCTS and by the temperature τ=1
# TODO: change arena: simpler MCTS, less iterations
arena = AlphaZero.ArenaParams(
  num_games=200,
  update_threshold=(2 * 0.52 - 1),
  mcts = mcts)

learning = AlphaZero.LearningParams(
  learning_rate=1e-3,
  epochs_per_checkpoint=10,
  max_num_epochs=50,
  stop_loss_eps=-1.0) # practically equivalent to -Inf

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  num_game_stages=5)

validation = AlphaZero.RolloutsValidation(
  num_games = 100,
  baseline = AlphaZero.MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0),
  contender = mcts)
