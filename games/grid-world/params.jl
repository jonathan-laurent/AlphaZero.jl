Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=100,
  depth_common=4,
  use_batch_norm=false)

self_play = SelfPlayParams(
  sim=SimParams(
    num_games=1000,
    num_workers=4,
    batch_size=4,
    use_gpu=false,
    reset_every=16,
    flip_probability=0.,
    alternate_colors=false),
  mcts=MctsParams(
    num_iters_per_turn=50,
    cpuct=1.0,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.,
    dirichlet_noise_α=1.))

arena = ArenaParams(
  sim=SimParams(
    num_games=100,
    num_workers=10,
    batch_size=10,
    use_gpu=false,
    reset_every=1,
    flip_probability=0.,
    alternate_colors=true),
  mcts = self_play.mcts,
  update_threshold=0.00)

learning = LearningParams(
  use_gpu=false,
  use_position_averaging=false,
  samples_weighing_policy=CONSTANT_WEIGHT,
  rewards_renormalization=10,
  l2_regularization=1e-4,
  optimiser=Adam(lr=5e-3),
  batch_size=64,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  min_checkpoints_per_epoch=1,
  max_batches_per_checkpoint=5_000,
  num_checkpoints=1)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=5,
  memory_analysis=nothing,
  ternary_rewards=false,
  use_symmetries=false,
  mem_buffer_size=PLSchedule(80_000))

benchmark_sim = SimParams(
  arena.sim;
  num_games=500,
  num_workers=10,
  batch_size=10)

benchmark = [
  Benchmark.Single(
    Benchmark.Full(self_play.mcts),
    benchmark_sim),
  Benchmark.Single(
    Benchmark.NetworkOnly(),
    benchmark_sim)]

experiment = Experiment(
  "grid-world", GameSpec(), params, Network, netparams, benchmark)