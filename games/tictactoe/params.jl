Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=500,
  depth_common=4)

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = AlphaZero.ArenaParams(
  num_games=3000,
  reset_mcts_every=3000,
  update_threshold=0.01,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))

self_play = AlphaZero.SelfPlayParams(
  num_games=2000,
  reset_mcts_every=2000,
  mcts = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=320,
    dirichlet_noise_ϵ=0.15))

learning = AlphaZero.LearningParams(
  l2_regularization=0.,
  nonvalidity_penalty=1.,
  max_num_epochs=40,
  first_checkpoint=8,
  stable_loss_n=15,
  stable_loss_ϵ=0.05)

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=3,
  num_game_stages=9)

validation = AlphaZero.RolloutsValidation(
  num_games = 1000,
  reset_mcts_every = 1000,
  baseline = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0.1),
  contender = AlphaZero.MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
