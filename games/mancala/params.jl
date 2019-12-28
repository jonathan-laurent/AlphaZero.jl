Network = ResNet

netparams = ResNetHP(
  num_filters=64,
  num_blocks=5,
  conv_kernel_size=(3,1),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.3)

self_play = SelfPlayParams(
  num_games=2_000,
  reset_mcts_every=400,
  mcts=MctsParams(
    use_gpu=true,
    num_workers=64,
    num_iters_per_turn=320,
    cpuct=4,
    temperature=StepSchedule(1.),
    dirichlet_noise_ϵ=0))

arena = ArenaParams(
  num_games=150,
  reset_mcts_every=100,
  update_threshold=(2 * 0.58 - 1),
  mcts=MctsParams(self_play.mcts,
    temperature=StepSchedule(0.3),
    dirichlet_noise_ϵ=0.05))

learning = LearningParams(
  batch_size=256,
  loss_computation_batch_size=1024,
  gc_every=nothing,
  learning_rate=1e-3,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2, 4])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=40,
  mem_buffer_size=PLSchedule(
    [      0,       20],
    [200_000, 2_000_000]))

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    num_games=100)]
