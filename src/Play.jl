#####
##### An MCTS-based player
#####

abstract type AbstractPlayer{Game} end

"""
    think(::AbstractPlayer, state, turn_number::Int)

Return an `(action, π)` pair where `action` is the chosen action and
`π` a probability distribution over available actions.

Note that `a` does not have to be drawn from `π`.
"""
function think(player::AbstractPlayer, state, turn)
  @unimplemented
end

function reset!(player::AbstractPlayer)
  return
end

#####
##### Random Player
#####

struct RandomPlayer{Game} <: AbstractPlayer{Game} end

function think(player::RandomPlayer, state, turn)
  actions = GI.available_actions(state)
  n = length(actions)
  π = ones(n) ./ length(actions)
  return π, rand(actions)
end

#####
##### MCTS with random oracle
#####

function RandomMctsPlayer(::Type{G}, params::MctsParams) where G
  oracle = MCTS.RandomOracle{G}()
  mcts = MCTS.Env{G}(oracle, 1, params.cpuct)
  return MctsPlayer(mcts, params.num_iters_per_turn,
    τ=params.temperature,
    nα=params.dirichlet_noise_nα,
    ϵ=params.dirichlet_noise_ϵ)
end

#####
##### MCTS Player
#####

struct MctsPlayer{G, M} <: AbstractPlayer{G}
  mcts :: M
  niters :: Int
  τ :: StepSchedule{Float64} # Temperature
  nα :: Float64 # Dirichlet noise parameter
  ϵ :: Float64 # Dirichlet noise weight
  function MctsPlayer(mcts::MCTS.Env{G}, niters; τ, nα, ϵ) where G
    new{G, typeof(mcts)}(mcts, niters, τ, nα, ϵ)
  end
end

# Alternative constructor
function MctsPlayer(oracle::MCTS.Oracle{G}, params::MctsParams) where G
  if isa(oracle, AbstractNetwork)
    oracle = Network.copy(oracle, on_gpu=params.use_gpu, test_mode=true)
  end
  mcts = MCTS.Env{G}(oracle, params.num_workers, params.cpuct)
  return MctsPlayer(mcts, params.num_iters_per_turn,
    τ=params.temperature,
    nα=params.dirichlet_noise_nα,
    ϵ=params.dirichlet_noise_ϵ)
end

function fix_probvec(π)
  π = convert(Vector{Float32}, π)
  s = sum(π)
  if !(s ≈ 1)
    if iszero(s)
      n = length(π)
      π = ones(Float32, n) ./ n
    else
      π ./= s
    end
  end
  return π
end

function think(p::MctsPlayer, state, turn)
  if iszero(p.niters)
    # Special case: use the oracle directly instead of MCTS
    actions = GI.available_actions(state)
    board = GI.canonical_board(state)
    π_mcts, _ = MCTS.evaluate(p.mcts.oracle, board, actions)
  else
    MCTS.explore!(p.mcts, state, p.niters)
    actions, π_mcts = MCTS.policy(p.mcts, state, τ=p.τ[turn])
  end
  if iszero(p.ϵ)
    π_exp = π_mcts
  else
    n = length(π_mcts)
    noise = Dirichlet(n, p.nα / n)
    π_exp = (1 - p.ϵ) * π_mcts + p.ϵ * rand(noise)
  end
  a = actions[rand(Categorical(fix_probvec(π_exp)))]
  return π_mcts, a
end

function reset!(player::MctsPlayer)
  MCTS.reset!(player.mcts)
end

#####
##### MCTS players can play against each other
#####

# Returns the reward and the game length
function play(
    white::AbstractPlayer{Game}, black::AbstractPlayer{Game}, memory=nothing
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
    π, a = think(player, state, nturns)
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

"""
    pit(handler, baseline, contender, ngames [, reset_period])

Evaluate two players against each other on a series of games,
alternating colors.

# Arguments

  - `handler`: this function is called after each simulated
     game with two arguments: the game number `i` and the collected reward `z`
     for the contender player
  - `baseline, contender :: AbstractPlayer`
  - `ngames`: number of games to play
  - `reset_period`: if set, players are reset every `reset_period` games
"""
function pit(
    handler, baseline::AbstractPlayer, contender::AbstractPlayer,
    ngames, reset_period=0)
  baseline_first = true
  zsum = 0.
  for i in 1:ngames
    white = baseline_first ? baseline : contender
    black = baseline_first ? contender : baseline
    z = play(white, black)
    baseline_first && (z = -z)
    zsum += z
    handler(i, z)
    if reset_period > 0 && (i % reset_period == 0 || i == ngames)
      reset!(baseline)
      reset!(contender)
    end
    baseline_first = !baseline_first
  end
  return zsum / ngames
end
