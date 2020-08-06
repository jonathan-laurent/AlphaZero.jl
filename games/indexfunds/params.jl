Network = SimpleNet

netparams = SimpleNetHP(
  width=200,
  depth_common=8,
  use_batch_norm=true,
  batch_norm_momentum=1.)

self_play = SelfPlayParams(
  num_games=10_000,
  mcts = MctsParams(
    num_workers=1,
    use_gpu=false,
    num_iters_per_turn=400,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  num_games=100,
  reset_mcts_every=1,
  update_threshold=0.00,
  flip_probability=0.5,
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1))

learning = LearningParams(
  use_gpu=false,
  samples_weighing_policy=LOG_WEIGHT,
  l2_regularization=1e-4,
  optimiser=CyclicNesterov(
    lr_base=1e-3,
    lr_high=1e-2,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.8),
  batch_size=32,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=0,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=2,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  ternary_rewards=true,
  use_symmetries=true,
  mem_buffer_size=PLSchedule(80_000))

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    num_games=400,
    flip_probability=0.5),
  Benchmark.Duel(
    Benchmark.NetworkOnly(),
    Benchmark.MinMaxTS(depth=6, amplify_rewards=true, τ=1.),
    num_games=400,
    flip_probability=0.5)]
