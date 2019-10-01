################################################################################
# MCTS player
################################################################################

using Distributions: Categorical, Dirichlet

################################################################################

struct MctsPlayer{M}
  mcts :: M
  niters :: Int
  τ  :: Float64 # Temperature
  nα :: Float64 # Dirichlet noise parameter
  ϵ  :: Float64 # Dirichlet noise weight
  MctsPlayer(mcts, niters; τ=1., nα=10., ϵ=0.) =
    new{typeof(mcts)}(mcts, niters, τ, nα, ϵ)
end

function think(p::MctsPlayer, state)
  MCTS.explore!(p.mcts, state, p.niters)
  actions, π_mcts = MCTS.policy(p.mcts, τ=p.τ)
  if iszero(p.ϵ)
    π_exp = π_mcts
  else
    n = length(π_mcts)
    noise = Dirichlet(n, p.nα / n)
    π_exp = (1 - p.ϵ) * π_mcts + p.ϵ * rand(noise)
  end
  a = actions[rand(Categorical(π_exp))]
  return π_mcts, a
end

################################################################################

function self_play!(Game, p::MctsPlayer, memory)
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      push_game!(memory, z)
      return
    end
    π, a = think(p, state)
    cboard = GI.canonical_board(state)
    push_sample!(memory, cboard, π, GI.white_playing(state))
    GI.play!(state, a)
  end
end

################################################################################
# Pitting Arena

function play_game(Game, white::MctsPlayer, black)
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z) return z :: Float64 end
    player = GI.white_playing(state) ? white : black
    π, a = think(player, state)
    GI.play!(state, a)
  end
end

################################################################################
