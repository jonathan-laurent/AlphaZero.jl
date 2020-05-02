Network = ResNet

netparams = ResNetHP(
  num_filters=64,
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play = SelfPlayParams(
  num_games=4_000,
  reset_mcts_every=100,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=32,
    num_iters_per_turn=600,
    cpuct=3.0,
    prior_temperature=0.7,
    temperature=PLSchedule([0, 20], [1.0, 0.3]),
    dirichlet_noise_ϵ=0.15,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  num_games=200,
  reset_mcts_every=nothing,
  flip_probability=0.5,
  update_threshold=0.05,
  mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.1),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=2048,
  loss_computation_batch_size=2048,
  optimiser=Adam(lr=1e-3),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=1000,
  num_checkpoints=2)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=50,
  ternary_rewards=true,
  use_symmetries=true,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  mem_buffer_size=PLSchedule(
  [      0,        60],
  [400_000, 2_000_000]))

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
    num_games=200,
    flip_probability=0.5,
    color_policy=CONTENDER_WHITE)

benchmark = make_duel.(baselines)
