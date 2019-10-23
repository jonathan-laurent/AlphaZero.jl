Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=600,
  depth_common=3,
  depth_vbranch=2,
  depth_pbranch=2)

self_play = AlphaZero.SelfPlayParams(
  num_games=600,
  reset_mcts_every=200,
  mcts=AlphaZero.MctsParams(
    use_gpu=true,
    num_workers=32,
    num_iters_per_turn=640,
    cpuct=4,
    temperature=1,
    dirichlet_noise_ϵ=0))

# Exploration is induced by MCTS and by the temperature τ=1
# TODO: change arena: simpler MCTS, less iterations
arena = AlphaZero.ArenaParams(
  num_games=1000,
  reset_mcts_every=1,
  update_threshold=(2 * 0.52 - 1),
  mcts=AlphaZero.MctsParams(
    use_gpu=true,
    num_workers=16,
    num_iters_per_turn=160,
    cpuct=4,
    temperature=0.3,
    dirichlet_noise_ϵ=0.05))

learning = AlphaZero.LearningParams(
  learning_rate=1e-3,
  l2_regularization=5e-6,
  nonvalidity_penalty=1.,
  checkpoints=[2, 4, 6, 12])

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
  contender=AlphaZero.MctsParams(
    use_gpu=true,
    num_workers=32,
    num_iters_per_turn=640,
    cpuct=4,
    temperature=0.3,
    dirichlet_noise_ϵ=0))
