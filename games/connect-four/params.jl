Network = NetLib.ResNet

netparams() = NetLib.ResNetHP(
  num_filters=128,
  num_blocks=5,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=32,
  num_value_head_filters=32,
  batch_norm_momentum=0.1)

self_play() = SelfPlayParams(
  sim=SimParams(
    num_games=5000,
    num_workers=128,
    use_gpu=true,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=600,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena() = ArenaParams(
  sim=SimParams(
    num_games=128,
    num_workers=128,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.5,
    alternate_colors=true),
  mcts=MctsParams(
    self_play().mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
  update_threshold=0.05)

learning(;l2_regularization=1e-4, lr=2e-3, batch_size=1024) = LearningParams(
  use_gpu=true,
  use_position_averaging=true,
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=batch_size,
  loss_computation_batch_size=1024,
  optimiser=Adam(lr=lr),
  l2_regularization=l2_regularization,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=2000,
  num_checkpoints=1)

params(;num_iters=20, kwargs...) = Params(
  arena=arena(),
  self_play=self_play(),
  learning=learning(;kwargs...),
  num_iters=num_iters,
  ternary_rewards=true,
  use_symmetries=true,
  memory_analysis=nothing,
  mem_buffer_size=PLSchedule(
  [      0,        40],
  [400_000, 2_000_000]))

mcts_baseline() =
  Benchmark.MctsRollouts(
    MctsParams(
      arena().mcts,
      num_iters_per_turn=1000,
      cpuct=1.))

minmax_baseline() = Benchmark.MinMaxTS(
  depth=5,
  τ=0.2,
  amplify_rewards=true)

players() = [
  Benchmark.Full(arena().mcts),
  Benchmark.NetworkOnly(τ=0.5)]

baselines() = [
  mcts_baseline(),
  mcts_baseline()]

benchmark_sim() = SimParams(
  arena().sim;
  num_games=256,
  num_workers=256,
  alternate_colors=false)

benchmark() = [
  Benchmark.Duel(p, b, benchmark_sim())
  for (p, b) in zip(players(), baselines())]

experiment() = Experiment(
  "connect-four",
  GameSpec(),
  params(;num_iters=20, l2_regularization=1e-4, lr=5e-3, batch_size=64),
  Network,
  netparams(),
  benchmark=benchmark(),
  )
