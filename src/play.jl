#####
##### Interface for players
#####

"""
    AbstractPlayer{Game}

Abstract type for a player of `Game`.
"""
abstract type AbstractPlayer{Game} end

"""
    think(::AbstractPlayer, state, turn=nothing)

Return a probability distribution over actions as a `(actions, π)` pair.

The `turn` argument, if provided, indicates the number of actions that have
been played before by both players in the current game.
It is useful as during self-play, AlphaZero typically drops its temperature
parameter after a fixed number of turns.
"""
function think(::AbstractPlayer, state, turn=nothing)
  @unimplemented
end

"""
    reset_player!(::AbstractPlayer)

Reset the internal memory of a player (e.g. the MCTS tree).
The default implementation does nothing.
"""
function reset_player!(::AbstractPlayer)
  return
end

"""
    select_move(player::AbstractPlayer, state, turn=nothing)

Return a single action. A default implementation is provided that samples
an action according to the distribution computed by [`think`](@ref).
"""
function select_move(player::AbstractPlayer, state, turn=nothing)
  actions, π = think(player, state, turn)
  return actions[Util.rand_categorical(π)]
end

GameType(::AbstractPlayer{Game}) where Game = Game

#####
##### Random Player
#####

"""
    RandomPlayer{Game} <: AbstractPlayer{Game}

A player that picks actions uniformly at random.
"""
struct RandomPlayer{Game} <: AbstractPlayer{Game} end

function think(player::RandomPlayer, state, turn=nothing)
  actions = GI.available_actions(state)
  n = length(actions)
  π = ones(n) ./ length(actions)
  return actions, π
end

#####
##### Epsilon-greedy player
#####

"""
    EpsilonGreedyPlayer{Game, Player} <: AbstractPlayer{Game}

A wrapper on a player that makes it choose a random move
with a fixed ``ϵ`` probability.
"""
struct EpsilonGreedyPlayer{G, P} <: AbstractPlayer{G}
  player :: P
  ϵ :: Float64
  function EpsilonGreedyPlayer(p::AbstractPlayer{G}, ϵ) where G
    return new{G, typeof(p)}(p, ϵ)
  end
end

function think(p::EpsilonGreedyPlayer, state, turn=nothing)
  actions, π = think(p.player, state, turn)
  n = length(actions)
  η = ones(n) ./ n
  return actions, (1 - p.ϵ) * π + p.ϵ * η
end

function reset!(p::EpsilonGreedyPlayer)
  reset!(p.player)
end

#####
##### An MCTS-based player
#####

"""
    MctsPlayer{Game, MctsEnv} <: AbstractPlayer{Game}

A player that selects actions using MCTS.

# Constructors

    MctsPlayer(mcts::MCTS.Env; τ, niters, timeout=nothing)

Construct a player from an MCTS environment. When computing each move:

- if `timeout` is provided,
  MCTS simulations are executed for `timeout` seconds by groups of `niters`
- otherwise, `niters` MCTS simulations are run

The temperature parameter `τ` can be either a real number or a
[`StepSchedule`](@ref).

    MctsPlayer(oracle::MCTS.Oracle, params::MctsParams; timeout=nothing)

Construct an MCTS player from an oracle and an [`MctsParams`](@ref) structure.
If the oracle is a network, this constructor handles copying it, putting it
in test mode and copying it on the GPU (if necessary).
"""
struct MctsPlayer{G, M} <: AbstractPlayer{G}
  mcts :: M
  niters :: Int
  timeout :: Union{Float64, Nothing}
  τ :: StepSchedule{Float64} # Temperature
  function MctsPlayer(mcts::MCTS.Env{G}; τ, niters, timeout=nothing) where G
    @assert niters > 0
    @assert isnothing(timeout) || timeout > 0
    if isa(τ, Number)
      τ = StepSchedule(Float64(τ))
    else
      @assert isa(τ, StepSchedule)
    end
    new{G, typeof(mcts)}(mcts, niters, timeout, τ)
  end
end

# Alternative constructor
function MctsPlayer(
    oracle::MCTS.Oracle{G}, params::MctsParams; timeout=nothing) where G
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
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature,
    timeout=timeout)
end

# MCTS with random oracle
function RandomMctsPlayer(::Type{G}, params::MctsParams) where G
  oracle = MCTS.RandomOracle{G}()
  mcts = MCTS.Env{G}(oracle,
    nworkers=1,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α)
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature)
end

function think(p::MctsPlayer, state, turn)
  if isnothing(p.timeout) # Fixed number of MCTS simulations
    MCTS.explore!(p.mcts, state, p.niters)
  else # Run simulations until timeout
    start = time()
    while time() - start < p.timeout
      MCTS.explore!(p.mcts, state, p.niters)
    end
  end
  return MCTS.policy(p.mcts, state, τ=p.τ[turn])
end

function reset_player!(player::MctsPlayer)
  MCTS.reset!(player.mcts)
end

#####
##### Network player
#####

"""
    NetworkPlayer{Game, Net} <: AbstractPlayer{Game}

A player that uses the policy output by a neural network directly,
instead of relying on MCTS.
"""
struct NetworkPlayer{G, N} <: AbstractPlayer{G}
  network :: N
  function NetworkPlayer(nn::AbstractNetwork{G}; use_gpu=true) where G
    nn = Network.copy(nn, on_gpu=use_gpu, test_mode=true)
    return new{G, typeof(nn)}(nn)
  end
end

function think(p::NetworkPlayer, state, turn)
  actions = GI.available_actions(state)
  board = GI.canonical_board(state)
  π, _ = MCTS.evaluate(p.network, board)
  return actions, π
end

#####
##### Players can play against each other
#####

"""
    play_game(white, black, memory=nothing)

Play a game between two [`AbstractPlayer`](@ref) and return the reward
obtained by `white`.

- If the `memory` argument is provided, samples are automatically collected
  from this game (see [`MemoryBuffer`](@ref)).
- If the `flip_probability` argument is set to ``p``, the board
  is _flipped_ randomly at every turn with probability ``p``,
  using [`GI.random_symmetric_state`](@ref).
"""
function play_game(
    white::AbstractPlayer{Game}, black::AbstractPlayer{Game},
    memory=nothing; flip_probability=0.) :: Float64 where Game
  state = Game()
  nturns = 0
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      isnothing(memory) || push_game!(memory, z, nturns)
      return z
    end
    if !iszero(flip_probability) && rand() < flip_probability
      state = GI.random_symmetric_state(state)
    end
    player = GI.white_playing(state) ? white : black
    actions, π = think(player, state, nturns)
    a = actions[Util.rand_categorical(π)]
    if !isnothing(memory)
      cboard = GI.canonical_board(state)
      push_sample!(memory, cboard, π, GI.white_playing(state), nturns)
    end
    GI.play!(state, a)
    nturns += 1
  end
end

#####
##### Evaluate two players against each other
#####

"""
    @enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

Policy for attributing colors in a duel between a baseline and a contender.
"""
@enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

"""
    pit(handler, contender, baseline, ngames)

Evaluate two `AbstractPlayer` against each other in a series of games.

# Arguments

  - `handler`: this function is called after each simulated
     game with two arguments: the game number `i` and the collected reward `z`
     for the contender player
  - `ngames`: number of games to play

# Optional keyword arguments
  - `reset_every`: if set, players are reset every `reset_every` games
  - `color_policy`: determines the [`ColorPolicy`](@ref),
     which is `ALTERNATE_COLORS` by default
  - `memory=nothing`: memory to use to record samples
  - `flip_probability=0.`: see [`play_game`](@ref)
"""
function pit(
    handler, contender::AbstractPlayer, baseline::AbstractPlayer, num_games;
    reset_every=nothing, color_policy=ALTERNATE_COLORS,
    memory=nothing, flip_probability=0.)
  baseline_white = (color_policy != CONTENDER_WHITE)
  zsum = 0.
  for i in 1:num_games
    white = baseline_white ? baseline : contender
    black = baseline_white ? contender : baseline
    z = play_game(white, black, memory, flip_probability=flip_probability)
    baseline_white && (z = -z)
    zsum += z
    handler(i, z)
    if !isnothing(reset_every) && (i % reset_every == 0 || i == num_games)
      reset_player!(baseline)
      reset_player!(contender)
    end
    if color_policy == ALTERNATE_COLORS
      baseline_white = !baseline_white
    end
  end
  return zsum / num_games
end

#####
##### Redudancy analysis
#####

# This type implements the interface of `MemoryBuffer` so that it can be
# passed as the `memory` argument of `play_game` to record games.
# Currently, this is only used to compute redundancy statistics
struct Recorder{Game, Board}
  boards :: Vector{Board}
  function Recorder{G}() where G
    B = GI.Board(G)
    return new{G, B}([])
  end
end

push_game!(r::Recorder, wr, gl) = nothing
push_sample!(r::Recorder, b, π, wp, t) = push!(r.boards, b)

function compute_redundancy(rec::Recorder{Game}) where Game
  initb = GI.board(Game())
  noninit = filter(!=(initb), rec.boards)
  unique = Set(noninit)
  return 1. - length(unique) / length(noninit)
end

#####
##### Simple utilities to play an interactive game
#####

"""
    Human{Game} <: AbstractPlayer{Game}

Human player that queries the standard input for actions.

Does not implement [`think`](@ref) but instead implements
[`select_move`](@ref) directly.
"""
struct Human{Game} <: AbstractPlayer{Game} end

struct Quit <: Exception end

function select_move(::Human, game, turn=nothing)
  a = nothing
  while isnothing(a) || a ∉ GI.available_actions(game)
    print("> ")
    str = readline()
    print("\n")
    isempty(str) && throw(Quit())
    a = GI.parse_action(game, str)
  end
  return a
end

"""
    interactive!(game, white, black)

Launch an interactive session for `game::AbstractGame` between players
`white` and `black`. Both players have type `AbstractPlayer` and one of them
is typically [`Human`](@ref).
"""
function interactive!(game, white, black)
  try
  GI.print_state(game)
  turn = 0
  while isnothing(GI.white_reward(game))
    player = GI.white_playing(game) ? white : black
    action = select_move(player, game, turn)
    GI.play!(game, action)
    GI.print_state(game)
    turn += 1
  end
  catch e
    isa(e, Quit) || rethrow(e)
    return
  end
end

interactive!(game::G) where G = interactive!(game, Human{G}(), Human{G}())
