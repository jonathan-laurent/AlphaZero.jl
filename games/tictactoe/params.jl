#####
##### Training regimes
#####

const USE_RESNET = true
const LEARNING_MODE = :fast

function get_params(mode=:full)
  @assert mode ∈ [:debug, :fast, :full]

  if mode == :debug
    num_iters = 3
    sp_num_games = 2
    sp_num_mcts_iters = 2
    arena_num_games = 2
    arena_win_rate_thresh = 0.51
    learning_batch_size = 2
    learning_checkpoints = [1]
    mem_buffer_size = PLSchedule(
      [  0,    4],
      [500, 2500])
    benchmark_num_games = 2

  elseif mode == :fast
    num_iters = 8
    sp_num_games = 100
    sp_num_mcts_iters = 20
    arena_num_games = 100
    arena_win_rate_thresh = 0.51
    learning_batch_size = 32
    learning_checkpoints = [10, 20]
    mem_buffer_size = PLSchedule(
      [  0,    4],
      [500, 2500])
    benchmark_num_games = 100

  else
    num_iters = 4
    sp_num_games = 4000
    sp_num_mcts_iters = 320
    arena_num_games = 1000
    arena_win_rate_thresh = 0.55
    learning_batch_size = 256
    learning_checkpoints = [1, 2, 5, 10, 20]
    mem_buffer_size = PLSchedule(
      [     0,      4],
      [20_000, 60_000])
    benchmark_num_games = 500
  end

  # Now we can build the parameters

  #####
  ##### Network parameters
  #####

  if USE_RESNET
    Net = ResNet{Game}
    netparams = ResNetHP(
      num_filters=64,
      num_blocks=5,
      conv_kernel_size=(3,3),
      num_policy_head_filters=32,
      num_value_head_filters=32,
      batch_norm_momentum=0.5)
  else
    Net = SimpleNet{Game}
    netparams = SimpleNetHP(
      width=500,
      depth_common=4)
  end

  #####
  ##### Training parameters
  #####

  self_play = SelfPlayParams(
    num_games=sp_num_games,
    reset_mcts_every=sp_num_games,
    mcts = MctsParams(
      num_workers=1,
      num_iters_per_turn=sp_num_mcts_iters,
      dirichlet_noise_ϵ=0.15))

  # Evaluate with 0 MCTS iterations
  # Exploration is induced by MCTS and by the temperature τ=1
  arena = ArenaParams(
    num_games=arena_num_games,
    reset_mcts_every=1,
    update_threshold=(2*arena_win_rate_thresh - 1),
    mcts = MctsParams(
      num_workers=1,
      num_iters_per_turn=0,
      dirichlet_noise_ϵ=0.1))

  learning = LearningParams(
    l2_regularization=1e-4,
    batch_size=learning_batch_size,
    loss_computation_batch_size=2048,
    nonvalidity_penalty=1.,
    checkpoints=learning_checkpoints)

  params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=num_iters,
    num_game_stages=9,
    mem_buffer_size=mem_buffer_size)

  benchmark = [
    Benchmark.Duel(
      Benchmark.Full(self_play.mcts),
      Benchmark.MctsRollouts(self_play.mcts),
      num_games=benchmark_num_games),
    Benchmark.Duel(
      Benchmark.NetworkOnly(self_play.mcts),
      Benchmark.MinMaxTS(depth=5),
      num_games=benchmark_num_games)]

  return Net, netparams, params, benchmark
end

Network, netparams, params, benchmark = get_params(LEARNING_MODE)
