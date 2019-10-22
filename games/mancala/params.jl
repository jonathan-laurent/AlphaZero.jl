Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=600,
  depth_common=3,
  depth_vbranch=2,
  depth_pbranch=2)

mcts = AlphaZero.MctsParams(
  num_workers=32,
  use_gpu=true,
  num_iters_per_turn=640,
  dirichlet_noise_ϵ=0)

self_play = AlphaZero.SelfPlayParams(
  num_games=600,
  reset_mcts_every=200,
  mcts=mcts)

# Exploration is induced by MCTS and by the temperature τ=1
# TODO: change arena: simpler MCTS, less iterations
arena = AlphaZero.ArenaParams(
  num_games=200,
  reset_mcts_every=40,
  update_threshold=(2 * 0.52 - 1),
  mcts=mcts)

learning = AlphaZero.LearningParams(
  learning_rate=1e-3,
  l2_regularization=1e-5,
  nonvalidity_penalty=1.,
  checkpoints=[2, 5, 20, 60])

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  num_game_stages=5)

validation = AlphaZero.RolloutsValidation(
  num_games=100,
  reset_mcts_every=20,
  baseline=AlphaZero.MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0),
  contender=mcts)
