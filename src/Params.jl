#####
##### AlphaZero Parameters
#####

# The default values are inspired by:
# https://github.com/suragnair/alpha-zero-general/blob/master/main.py
# For dirichlet noise parameters, see:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3

@kwdef struct MctsParams
  num_workers :: Int = 1
  use_gpu :: Bool = false
  cpuct :: Float64 = 1.
  num_iters_per_turn :: Int
  temperature :: Float64 = 1.
  dirichlet_noise_nα :: Float64 = 10.
  dirichlet_noise_ϵ :: Float64 = 0.
end

@kwdef struct ArenaParams
  num_games :: Int
  reset_mcts_every :: Int
  mcts :: MctsParams
  update_threshold :: Float64
end

@kwdef struct SelfPlayParams
  num_games :: Int
  reset_mcts_every :: Int
  mcts :: MctsParams
end

@kwdef struct LearningParams
  use_gpu :: Bool = true
  # Optimization parameters
  learning_rate :: Float32 = 1e-3
  l2_regularization :: Float32 = 0.
  nonvalidity_penalty :: Float32 = 1.
  batch_size :: Int = 32
  # Checkpoint parameters
  max_num_epochs :: Int
  first_checkpoint :: Int
  stable_loss_ϵ :: Float64
  stable_loss_n :: Int
end

@kwdef struct Params
  arena :: ArenaParams
  self_play :: SelfPlayParams
  learning :: LearningParams
  num_iters :: Int
  mem_buffer_size :: Int = 200_000
  num_game_stages :: Int # as featured in memory reports
end

#####
##### Parameters from the original AlphaZero paper
#####

# Alpha Zero dev
# + 4.9 millions games of self play
# + Parameters updated from 700,000 minibatches of 2048 positions
# + Neural network: 20 residual blocks
# + 1600 MCTS iterations per turn (0.4s)
# + Checkpoint after 1000 training steps (2048x1000/500K ≃ 4 epochs)
# + First 30 moves, τ=1, then τ → 0 (not implemented here)
# Alpha Zero final
# + 29 million games, 31 million minibatches of 2048 positions
#=
const ALPHA_ZERO_PAPER_PARAMS = Params(
  arena = ArenaParams(
    num_games = 400,
    num_mcts_iters_per_turn = 1600, #0.4s
    update_threshold = (0.55 + 1) / 2
  ),
  self_play = SelfPlayParams(
    num_games = 25_000,
    num_mcts_iters_per_turn = 1600,
    dirichlet_noise_nα = 10.,
    dirichlet_noise_ϵ = 0.25
  ),
  learning = LearningParams(
    batch_size = 2048,
  ),
  num_iters = 200, # 5M / 25K
  mem_buffer_size = 500_000,
)
=#
