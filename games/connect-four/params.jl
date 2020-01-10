const DEBUG = get(ENV, "TRAINING_MODE", "") == "debug"

const COLD_TEMPERATURE = 0.

Network = ResNet

netparams = ResNetHP(
  num_filters=128,
  num_blocks=8,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.6)

self_play = SelfPlayParams(
  num_games=(DEBUG ? 1 : 5_000),
  reset_mcts_every=1_000,
  gc_every=nothing,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=400,
    cpuct=3,
    temperature=StepSchedule(
      start=1.0,
      change_at=[8],
      values=[COLD_TEMPERATURE]),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  num_games=(DEBUG ? 1 : 200),
  reset_mcts_every=nothing,
  update_threshold=(2 * 0.55 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(COLD_TEMPERATURE)))

learning = LearningParams(
  samples_weighing_policy=LOG_WEIGHT,
  batch_size=512,
  loss_computation_batch_size=1024,
  gc_every=nothing,
  optimiser=CyclicMomentum(
    lr_base=1e-2,
    lr_high=1e-1,
    lr_low=1e-3,
    momentum_high=0.9,
    momentum_low=0.7),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=120,
  ternary_rewards=true,
  memory_analysis=MemAnalysisParams(
    num_game_stages=4),
  mem_buffer_size=PLSchedule(
  [      0,        40],
  [200_000, 2_000_000]))

deployed_mcts = MctsParams(self_play.mcts,
  temperature=StepSchedule(COLD_TEMPERATURE))

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MctsRollouts(
      MctsParams(deployed_mcts, num_iters_per_turn=1000)),
    num_games=(DEBUG ? 1 : 200),
    color_policy=CONTENDER_WHITE),
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MinMaxTS(depth=5, τ=0.2),
    num_games=(DEBUG ? 1 : 200),
    color_policy=CONTENDER_WHITE),
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.Solver(ϵ=0.1),
    num_games=(DEBUG ? 1 : 200),
    color_policy=CONTENDER_WHITE)]
