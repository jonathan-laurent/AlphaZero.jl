const RESNET = true

if RESNET
  Network = ResNet{Game}
  netparams = ResNetHP(
    num_filters=64,
    num_blocks=5,
    conv_kernel_size=(3,3),
    num_policy_head_filters=32,
    num_value_head_filters=32,
    batch_norm_momentum=0.5)
else
  Network = SimpleNet{Game}
  netparams = SimpleNetHP(
    width=300,
    depth_common=3)
end

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = ArenaParams(
  num_games=400,
  reset_mcts_every=50,
  update_threshold=(2 * 0.51 - 1),
  mcts = MctsParams(
    num_iters_per_turn=0))

self_play = SelfPlayParams(
  num_games=100,
  reset_mcts_every=50,
  mcts = MctsParams(
    num_workers=1,
    use_gpu=true,
    num_iters_per_turn=20,
    dirichlet_noise_ϵ=0.15))

learning = LearningParams(
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  batch_size=32,
  loss_computation_batch_size=512,
  checkpoints=[10, 20])

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=8,
  num_game_stages=9,
  mem_buffer_size=PLSchedule(
    [  0,    4],
    [500, 2500]))

validation = RolloutsValidation(
  num_games = 100,
  reset_mcts_every=100,
  baseline = MctsParams(
    num_iters_per_turn=100,
    dirichlet_noise_ϵ=0.1),
  contender = MctsParams(
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))
