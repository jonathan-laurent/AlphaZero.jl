Network = ResNet

netparams = ResNetHP(
  num_filters=64,
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play = SelfPlayParams(
  num_games=4_000,
  reset_mcts_every=1,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=320,
    cpuct=2.0,
    temperature=StepSchedule(
      start=1.0,
      change_at=[10],
      values=[0.4]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  num_games=200,
  reset_mcts_every=nothing,
  flip_probability=0.5,
  update_threshold=0.10,
  mcts=MctsParams(
    self_play.mcts,
    temperature=StepSchedule(0.1),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=1024,
  loss_computation_batch_size=1024,
  optimiser=CyclicNesterov(
    lr_base=1e-2,
    lr_high=1e-1,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=1000,
  num_checkpoints=2)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=120,
  ternary_rewards=true,
  use_symmetries=true,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  mem_buffer_size=PLSchedule(
  [      0,        60],
  [360_000, 1_800_000]))

baselines = [
  Benchmark.MctsRollouts(
    MctsParams(
      arena.mcts,
      num_iters_per_turn=1000,
      cpuct=1.)),
  Benchmark.MinMaxTS(depth=5, τ=0.2)]

# push!(baselines, Benchmark.Solver(ϵ=0.05))

make_duel(baseline) =
  Benchmark.Duel(
    Benchmark.Full(arena.mcts),
    baseline,
    num_games=100,
    flip_probability=0.5,
    color_policy=CONTENDER_WHITE)

benchmark = make_duel.(baselines)
