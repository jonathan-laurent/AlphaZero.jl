#####
##### An MCTS-based player
#####

struct MctsPlayer{G, M}
  mcts :: M
  niters :: Int
  τ :: Float64 # Temperature
  nα :: Float64 # Dirichlet noise parameter
  ϵ :: Float64 # Dirichlet noise weight
  function MctsPlayer(mcts::MCTS.Env{G}, niters; τ=1., nα=10., ϵ=0.) where G
    new{G, typeof(mcts)}(mcts, niters, τ, nα, ϵ)
  end
end

# Alternative constructor
function MctsPlayer(oracle::MCTS.Oracle{G}, params::MctsParams) where G
  if isa(oracle, AbstractNetwork)
    oracle = params.use_gpu ?
      Network.to_gpu(oracle) :
      Network.to_cpu(oracle)
  end
  mcts = MCTS.Env{G}(oracle, params.num_workers, params.cpuct)
  return MctsPlayer(mcts, params.num_iters_per_turn,
    τ=params.temperature,
    nα=params.dirichlet_noise_nα,
    ϵ=params.dirichlet_noise_ϵ)
end

function think(p::MctsPlayer, state)
  if iszero(p.niters)
    # Special case: use the oracle directly instead of MCTS
    actions = GI.available_actions(state)
    board = GI.canonical_board(state)
    π_mcts, _ = MCTS.evaluate(p.mcts.oracle, board, actions)
  else
    MCTS.explore!(p.mcts, state, p.niters)
    actions, π_mcts = MCTS.policy(p.mcts, state, τ=p.τ)
  end
  if iszero(p.ϵ)
    π_exp = π_mcts
  else
    n = length(π_mcts)
    noise = Dirichlet(n, p.nα / n)
    π_exp = (1 - p.ϵ) * π_mcts + p.ϵ * rand(noise)
  end
  # The line below is necessary so that Distributions.isprob
  # does not get too picky.
  π_exp = convert(Vector{Float32}, π_exp)
  a = actions[rand(Categorical(π_exp))]
  return π_mcts, a
end

#####
##### MCTS players can play against each other
#####

# Returns the reward and the game length
function play(
    white::MctsPlayer{Game}, black::MctsPlayer{Game}, memory=nothing
  ) :: Float64 where Game
  state = Game()
  nturns = 0
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      isnothing(memory) || push_game!(memory, z, nturns)
      return z
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

self_play!(player, memory) = play(player, player, memory)

#####
##### Evaluate two players against each other
#####

function pit(
    handler, baseline::MctsPlayer, contender::MctsPlayer, ngames, reset_period)
  baseline_first = true
  zsum = 0.
  for i in 1:ngames
    white = baseline_first ? baseline : contender
    black = baseline_first ? contender : baseline
    z = play(white, black)
    baseline_first && (z = -z)
    zsum += z
    handler(i, z)
    baseline_first = !baseline_first
    if i % reset_period == 0 || i == ngames
      MCTS.reset!(baseline.mcts)
      MCTS.reset!(contender.mcts)
    end
  end
  return zsum / ngames
end

function pit(baseline, contender, ngames, reset_period)
  return pit((i, z) -> nothing, baseline, contender, ngames, reset_period)
end
