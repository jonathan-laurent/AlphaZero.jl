#####
##### Training regimes
#####

const DEBUG = true

const FAST_TRAINING = true

const USE_RESNET = true

if DEBUG

  NUM_ITERS = 3

  SP_NUM_GAMES   = 2
  SP_NUM_MCTS_ITERS = 2

  ARENA_NUM_GAMES = 2
  ARENA_WIN_RATE_THRESH = 0.51

  LEARNING_BATCH_SIZE = 2
  LEARNING_CHECKPOINTS = [1]

  MEM_BUFFER_SIZE = PLSchedule(
    [  0,    4],
    [500, 2500])

  VALIDATION_NUM_GAMES = 2

elseif FAST_TRAINING

  NUM_ITERS = 8

  SP_NUM_GAMES   = 100
  SP_NUM_MCTS_ITERS = 20

  ARENA_NUM_GAMES = 100
  ARENA_WIN_RATE_THRESH = 0.51

  LEARNING_BATCH_SIZE = 32
  LEARNING_CHECKPOINTS = [10, 20]

  MEM_BUFFER_SIZE = PLSchedule(
    [  0,    4],
    [500, 2500])

  VALIDATION_NUM_GAMES = 100

else # Long training

  NUM_ITERS = 4

  SP_NUM_GAMES   = 4000
  NUM_MCTS_ITERS = 320

  ARENA_NUM_GAMES = 1000
  ARENA_WIN_RATE_THRESH = 0.55

  LEARNING_BATCH_SIZE = 256
  LEARNING_CHECKPOINTS = [1, 2, 5, 10, 20]

  MEM_BUFFER_SIZE = PLSchedule(
    [     0,      4],
    [20_000, 60_000])

  VALIDATION_NUM_GAMES = 500

end

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
  num_games=SP_NUM_GAMES,
  reset_mcts_every=SP_NUM_GAMES,
  mcts = MctsParams(
    num_workers=1,
    num_iters_per_turn=SP_NUM_MCTS_ITERS,
    dirichlet_noise_ϵ=0.15))

# Evaluate with 0 MCTS iterations
# Exploration is induced by MCTS and by the temperature τ=1
arena = ArenaParams(
  num_games=ARENA_NUM_GAMES,
  reset_mcts_every=1,
  update_threshold=(2*ARENA_WIN_RATE_THRESH - 1),
  mcts = MctsParams(
    num_workers=1,
    num_iters_per_turn=0,
    dirichlet_noise_ϵ=0.1))

learning = LearningParams(
  l2_regularization=1e-4,
  batch_size=256,
  loss_computation_batch_size=2048,
  nonvalidity_penalty=1.,
  checkpoints=LEARNING_CHECKPOINTS)

params = Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=NUM_ITERS,
  num_game_stages=9,
  mem_buffer_size=MEM_BUFFER_SIZE)

benchmark = [
  Benchmark.Duel(
    Benchmark.Full(self_play.mcts),
    Benchmark.MctsRollouts(self_play.mcts),
    num_games=VALIDATION_NUM_GAMES)]
