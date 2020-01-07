const DEBUG = get(ENV, "TRAINING_MODE", "") == "debug"

const SMALLER_NETWORK = true

cold_temperature = 0.05

Network = ResNet

netparams = ResNetHP(
  num_filters=(SMALLER_NETWORK ? 64 : 128),
  num_blocks=7,
  conv_kernel_size=(3, 3),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.3)

self_play = SelfPlayParams(
  num_games=(DEBUG ? 1 : 5_000),
  reset_mcts_every=1_000,
  gc_every=nothing,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=320,
    cpuct=3,
    temperature=StepSchedule(
      start=1.0,
      change_at=[8],
      values=[cold_temperature]),
    dirichlet_noise_ϵ=0.2,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
  num_games=(DEBUG ? 1 : 400),
  reset_mcts_every=nothing,
  update_threshold=(2 * 0.54 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(cold_temperature)))

learning = LearningParams(
  batch_size=256,
  loss_computation_batch_size=1024,
  gc_every=nothing,
  optimiser=CyclicMomentum(
    lr_low=0.01,
    lr_high=0.1,
    momentum_high=0.9,
    momentum_low=0.7),
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  memory_analysis=nothing,
  num_iters=120,
  ternary_rewards=true,
  mem_buffer_size=PLSchedule(
  [      0,        20],
  [150_000, 1_000_000]))

deployed_mcts = MctsParams(self_play.mcts,
  temperature=StepSchedule(cold_temperature),
  dirichlet_noise_ϵ=0.1)

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MctsRollouts(
      MctsParams(deployed_mcts, num_iters_per_turn=1000)),
    num_games=(DEBUG ? 1 : 400),
    color_policy=CONTENDER_WHITE),
  Benchmark.Duel(
    Benchmark.Full(deployed_mcts),
    Benchmark.MinMaxTS(depth=4, τ=0.2),
    num_games=(DEBUG ? 1 : 400),
    color_policy=CONTENDER_WHITE)]
