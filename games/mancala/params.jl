Network = ResNet{Game}

netparams = ResNetHP(
  num_filters=64,
  num_blocks=7,
  conv_kernel_size=(3,1),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.5)

self_play = SelfPlayParams(
  num_games=600,
  reset_mcts_every=200,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=32,
    num_iters_per_turn=640,
    cpuct=4,
    temperature=1,
    dirichlet_noise_ϵ=0))

arena = ArenaParams(
  num_games=800,
  reset_mcts_every=20,
  update_threshold=(2 * 0.53 - 1),
  mcts=MctsParams(self_play.mcts,
    num_workers=16,
    num_iters_per_turn=160,
    temperature=0.3,
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  batch_size=256,
  loss_computation_batch_size=2048,
  learning_rate=1e-3,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[2, 4, 6, 12])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  num_game_stages=5,
  mem_buffer_size=PLSchedule(
    [      0,      20],
    [100_000, 500_000]))

validation = RolloutsValidation(
  num_games=100,
  reset_mcts_every=20,
  baseline=MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0),
  contender=MctsParams(self_play.mcts,
    temperature=0.3,
    dirichlet_noise_ϵ=0))
