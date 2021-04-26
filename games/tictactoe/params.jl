Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=200,
  depth_common=6,
  use_batch_norm=true,
  batch_norm_momentum=1.)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=1000,
    num_workers=128,
    batch_size=128,
    use_gpu=false,
    reset_every=4,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=400,
    cpuct=1.0,
    temperature=ConstSchedule(1.0),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  sim=SimParams(
    num_games=100,
    num_workers=100,
    batch_size=100,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.5,
    alternate_colors=true),
  mcts = MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.3),
    dirichlet_noise_ϵ=0.1),
  update_threshold=0.00)

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

benchmark_sim = SimParams(
  arena.sim;
  num_games=400,
  num_workers=100,
  batch_size=100)

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    benchmark_sim),
  Benchmark.Duel(
    Benchmark.NetworkOnly(),
    Benchmark.MinMaxTS(depth=6, amplify_rewards=true, τ=1.),
    benchmark_sim)]

experiment = Experiment(
  "tictactoe", GameSpec(), params, Network, netparams, benchmark)