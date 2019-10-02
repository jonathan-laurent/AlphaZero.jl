#####
##### An MCTS-based player
#####

struct MctsPlayer{M}
  mcts :: M
  niters :: Int
  τ :: Float64 # Temperature
  nα :: Float64 # Dirichlet noise parameter
  ϵ :: Float64 # Dirichlet noise weight
  function MctsPlayer(mcts, niters; τ=1., nα=10., ϵ=0.)
    new{typeof(mcts)}(mcts, niters, τ, nα, ϵ)
  end
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


#####
##### MCTS players can play against each other
#####

function play(
  ::Type{Game}, white::MctsPlayer, black::MctsPlayer, memory=nothing
  ) where Game
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      isnothing(memory) || push_game!(memory, z)
      return z
    end
    player = GI.white_playing(state) ? white : black
    π, a = think(player, state)
    if !isnothing(memory)
      cboard = GI.canonical_board(state)
      push_sample!(memory, cboard, π, GI.white_playing(state))
    end
    GI.play!(state, a)
  end
end

self_play!(G, player, memory) = play(G, player, player, memory)


#####
##### Evaluating the latest NN
##### by pitting it against the best one so far
#####

# Returns average reward for the evaluated player
#   Question: when pitting network against each other,
#   where does randomness come from?
#   Answer: we leave a nonzero temperature
function evaluate_oracle(
    ::Type{G}, baseline::Oracle{G}, oracle::Oracle{G}, params::ArenaParams
  ) where G
  τ = params.temperature
  n_mcts = params.num_mcts_iters_per_turn
  n_episodes = params.num_games
  best_mcts = MCTS.Env{G}(baseline, params.cpuct)
  new_mcts = MCTS.Env{G}(oracle, params.cpuct)
  best = MctsPlayer(best_mcts, n_mcts, τ=τ)
  new = MctsPlayer(new_mcts, n_mcts, τ=τ)
  zsum = 0.
  best_first = true
  for i in 1:n_episodes
    white = best_first ? best : new
    black = best_first ? new : best
    z = play(G, white, black)
    best_first && (z = -z)
    zsum += z
    best_first = !best_first
  end
  return zsum / n_episodes
end
