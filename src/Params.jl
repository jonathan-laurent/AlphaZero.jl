################################################################################
# AlphaZero Parameters
################################################################################

using Base: @kwdef

@kwdef struct ArenaParams
  num_mcts_iters_per_turn :: Int = 25
  num_games :: Int = 40
  temperature :: Float64 = 0.4
  update_threshold :: Float64 = 0.55 # 0.6 in Nair'simplementation
end

@kwdef struct SelfPlayParams
  num_mcts_iters_per_turn :: Int = 250
  temperature :: Float64 = 1.
  dirichlet_noise_nα :: Float64 = 10.
  dirichlet_noise_ϵ :: Float64 = 0.25
end

@kwdef struct LearningParams
  learning_rate :: Float64 = 1e-3
  num_batches :: Int = 10_000
  batch_size :: Int = 32
  loss_eps :: Float64 = 1e-3
end

@kwdef struct Params
  arena :: ArenaParams = ArenaParams()
  self_play :: SelfPlayParams = SelfPlayParams()
  learning :: LearningParams = LearningParams()
  num_learning_iters :: Int = 100
  num_episodes_per_iter :: Int = 25
  mem_buffer_size :: Int = 200_000
  cpuct :: Float64 = 1.
end

# Some standard values for params:
# https://github.com/suragnair/alpha-zero-general/blob/master/main.py
# For dirichlet noise, see:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3

################################################################################
# Alpha Zero dev
# + 4.9 millions games of self play
# + Parameters updated from 700,000 minibatches of 2048 positions
# + Neural network: 20 residual blocks
# + 1600 MCTS iterations per turn (0.4s)
# + Checkpoint after 1000 training steps
# + First 30 moves, τ=1, then τ → 0 (not implemented here)
# Alpha Zero final
# + 29 million games, 31 million minibatches of 2048 positions
################################################################################

const ALPHA_ZERO_PAPER_PARAMS = Params(
  arena = ArenaParams(
    num_games = 400,
    num_mcts_iters_per_turn = 1600, #0.4s
    update_threshold = 0.55
  ),
  self_play = SelfPlayParams(
    num_mcts_iters_per_turn = 1600,
    dirichlet_noise_nα = 10.,
    dirichlet_noise_ϵ = 0.25
  ),
  learning = LearningParams(
    batch_size = 2048,
    num_batches = 1000
  ),
  num_episodes_per_iter = 25_000,
  num_learning_iters = 200, # 5M / 25K
  mem_buffer_size = 500_000,
  cpuct = 1.
)

################################################################################
