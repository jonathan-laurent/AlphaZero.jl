#####
##### An MCTS-based player
#####

struct MctsPlayer{G, M} <: AbstractPlayer{G}
  mcts :: M
  niters :: Int
  τ :: StepSchedule{Float64} # Temperature
  function MctsPlayer(mcts::MCTS.Env{G}, niters; τ) where G
    @assert niters > 0
    new{G, typeof(mcts)}(mcts, niters, τ)
  end
end

# Alternative constructor
function MctsPlayer(oracle::MCTS.Oracle{G}, params::MctsParams) where G
  fill_batches = false
  if isa(oracle, AbstractNetwork)
    oracle = Network.copy(oracle, on_gpu=params.use_gpu, test_mode=true)
    params.use_gpu && (fill_batches = true)
  end
  mcts = MCTS.Env{G}(oracle,
    nworkers=params.num_workers,
    fill_batches=fill_batches,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α)
  return MctsPlayer(mcts, params.num_iters_per_turn, τ=params.temperature)
end

# MCTS with random oracle
function RandomMctsPlayer(::Type{G}, params::MctsParams) where G
  oracle = MCTS.RandomOracle{G}()
  mcts = MCTS.Env{G}(oracle,
    nworkers=1,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α)
  return MctsPlayer(mcts, params.num_iters_per_turn, τ=params.temperature)
end

function GI.think(p::MctsPlayer, state, turn)
  MCTS.explore!(p.mcts, state, p.niters)
  return MCTS.policy(p.mcts, state, τ=p.τ[turn])
end

function GI.reset_player!(player::MctsPlayer)
  MCTS.reset!(player.mcts)
end

#####
##### Network player
#####

struct NetworkPlayer{G, N} <: AbstractPlayer{G}
  network :: N
  function NetworkPlayer(nn::AbstractNetwork{G}; use_gpu=true) where G
    nn = Network.copy(nn, on_gpu=use_gpu, test_mode=true)
    return new{G, typeof(nn)}(nn)
  end
end

function GI.think(p::NetworkPlayer, state, turn)
  actions = GI.available_actions(state)
  board = GI.canonical_board(state)
  π, _ = MCTS.evaluate(p.network, board, actions)
  return actions, π
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
    actions, π = GI.think(player, state, nturns)
    a = actions[Util.rand_categorical(π)]
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
    @enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

Policy for attributing colors in a duel between a baseline and a contender.
"""
@enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

"""
    pit(handler, baseline, contender, ngames)

Evaluate two players against each other on a series of games.

# Arguments

  - `handler`: this function is called after each simulated
     game with two arguments: the game number `i` and the collected reward `z`
     for the contender player
  - `baseline, contender :: AbstractPlayer`
  - `ngames`: number of games to play

# Optional keyword arguments
  - `reset_every`: if set, players are reset every `reset_every` games
  - `color_policy`: determine the [`ColorPolicy`](@ref),
    which is `ALTERNATE_COLORS` by default
"""
function pit(
    handler, baseline::AbstractPlayer, contender::AbstractPlayer, num_games;
    reset_every=nothing, color_policy=ALTERNATE_COLORS)
  baseline_white = (color_policy != CONTENDER_WHITE)
  zsum = 0.
  for i in 1:num_games
    white = baseline_white ? baseline : contender
    black = baseline_white ? contender : baseline
    z = play(white, black)
    baseline_white && (z = -z)
    zsum += z
    handler(i, z)
    if !isnothing(reset_every) && (i % reset_every == 0 || i == num_games)
      GI.reset_player!(baseline)
      GI.reset_player!(contender)
    end
    if color_policy == ALTERNATE_COLORS
      baseline_white = !baseline_white
    end
  end
  return zsum / num_games
end
