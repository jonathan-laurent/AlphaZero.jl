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
  actions, π_mcts = MCTS.policy(p.mcts, state, τ=p.τ)
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

# Returns the reward and the game length
function play(
  ::Type{Game}, white::MctsPlayer, black::MctsPlayer, memory=nothing
  ) :: Report.Game where Game
  state = Game()
  nturns = 0
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      isnothing(memory) || push_game!(memory, z, nturns)
      return Report.Game(z, nturns)
    end
    player = GI.white_playing(state) ? white : black
    π, a = think(player, state)
    if !isnothing(memory)
      cboard = GI.canonical_board(state)
      push_sample!(memory, cboard, π, GI.white_playing(state), nturns)
    end
    GI.play!(state, a)
    nturns += 1
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
    baseline::MCTS.Oracle{G}, oracle::MCTS.Oracle{G}, params::ArenaParams
  ) :: Report.Evaluation where G
  τ = params.temperature
  n_mcts = params.num_mcts_iters_per_turn
  n_episodes = params.num_games
  best_mcts = MCTS.Env{G}(baseline, params.cpuct)
  new_mcts = MCTS.Env{G}(oracle, params.cpuct)
  best = MctsPlayer(best_mcts, n_mcts, τ=τ)
  new = MctsPlayer(new_mcts, n_mcts, τ=τ)
  zsum = 0.
  best_first = true
  games = Vector{Report.Game}(undef, n_episodes)
  for i in 1:n_episodes
    if params.reset_mcts
      MCTS.reset!(best_mcts)
      MCTS.reset!(new_mcts)
    end
    white = best_first ? best : new
    black = best_first ? new : best
    grep = play(G, white, black)
    if best_first
      grep = Report.Game(-grep.reward, grep.length)
    end
    games[i] = grep
    best_first = !best_first
  end
  avgz = mean(g.reward for g in games)
  return Report.Evaluation(games, avgz)
end
