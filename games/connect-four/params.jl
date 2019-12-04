const DEBUG = false

cold_temperature = 0.2

Net = ResNet{Game}

netparams = ResNetHP(
  num_filters=128,
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.3)

self_play = SelfPlayParams(
  num_games=(DEBUG ? 20 : 4_000),
  reset_mcts_every=1_000,
  gc_every=0,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=320,
    cpuct=3,
    temperature=StepSchedule(
      start=1.0,
      change_at=[10],
      values=[cold_temperature]),
    dirichlet_noise_ϵ=0.05))

arena = ArenaParams(
  num_games=(DEBUG ? 15 : 200),
  reset_mcts_every=200,
  update_threshold=(2 * 0.55 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  batch_size=256,
  loss_computation_batch_size=1024,
  gc_every=0,
  learning_rate=1e-3,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=100,
  num_game_stages=5,
  perform_memory_analysis=false,
  mem_buffer_size=PLSchedule(
  [      0,        20],
  [120_000, 1_000_000]))

validation = RolloutsValidation(
  num_games=(DEBUG ? 10 : 200),
  reset_mcts_every=200,
  baseline=MctsParams(
    num_iters_per_turn=1000,
    dirichlet_noise_ϵ=0),
  contender=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature),
    dirichlet_noise_ϵ=0))

# validation = nothing
