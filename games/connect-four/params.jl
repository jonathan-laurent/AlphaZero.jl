const DEBUG = get(ENV, "TRAINING_MODE", "") == "debug"

cold_temperature = 0.05

Network = ResNet{Game}

netparams = ResNetHP(
  num_filters=128,
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.3)

self_play = SelfPlayParams(
  num_games=(DEBUG ? 1 : 6_000),
  reset_mcts_every=1_000,
  gc_every=nothing,
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
  num_games=(DEBUG ? 1 : 400),
  reset_mcts_every=400,
  update_threshold=(2 * 0.54 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  batch_size=256,
  loss_computation_batch_size=1024,
  gc_every=nothing,
  learning_rate=2e-4,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  memory_analysis=nothing,
  num_iters=80,
  ternary_rewards=true,
  mem_buffer_size=PLSchedule(
  [      0,        20],
  [200_000, 1_000_000]))

deployed_mcts = MctsParams(self_play.mcts,
  temperature=StepSchedule(cold_temperature),
  dirichlet_noise_ϵ=0)

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MctsRollouts(
      MctsParams(deployed_mcts, num_iters_per_turn=1000)),
    num_games=(DEBUG ? 1 : 200),
    color_policy=CONTENDER_WHITE),
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MinMaxTS(depth=4, random_ϵ=0.05),
    num_games=(DEBUG ? 1 : 200),
    color_policy=CONTENDER_WHITE)]
