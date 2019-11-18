Network = ResNet{Game}

cold_temperature = 0.2

netparams = ResNetHP(
  num_filters=64,
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.05)

self_play = SelfPlayParams(
  num_games=3_000,
  reset_mcts_every=600,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=320,
    cpuct=4,
    temperature=StepSchedule(
      start=1.0,
      change_at=[10],
      values=[cold_temperature]),
    dirichlet_noise_ϵ=0.05))

arena = ArenaParams(
  num_games=150,
  reset_mcts_every=100,
  update_threshold=(2 * 0.58 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  batch_size=64,
  loss_computation_batch_size=64,
  gc_every=0,
  learning_rate=1e-3,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  num_game_stages=5,
  mem_buffer_size=PLSchedule(
    [      0,       40],
    [200_000, 2_000_000]))

validation = RolloutsValidation(
  num_games=200,
  reset_mcts_every=100,
  baseline=MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0),
  contender=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature),
    dirichlet_noise_ϵ=0))

# validation = nothing
