#####
##### Interface for players
#####

"""
    AbstractPlayer{Game}

Abstract type for a player of `Game`.
"""
abstract type AbstractPlayer{Game} end

GameType(::AbstractPlayer{Game}) where Game = Game

"""
    think(::AbstractPlayer, state)

Return a probability distribution over actions as a `(actions, π)` pair.
"""
function think end

"""
    reset_player!(::AbstractPlayer)

Reset the internal memory of a player (e.g. the MCTS tree).
The default implementation does nothing.
"""
function reset_player!(::AbstractPlayer)
  return
end

"""
    player_temperature(::AbstractPlayer, turn_number)

Return the player temperature, given the number of actions that have
been played before by both players in the current game.

A default implementation is provided that always returns 1.
"""
function player_temperature(::AbstractPlayer, turn)
  return 1.0
end

"""
    select_move(player::AbstractPlayer, state, turn_number)

Return a single action. A default implementation is provided that samples
an action according to the distribution computed by [`think`](@ref), with a
temperature given by [`player_temperature`](@ref).
"""
function select_move(player::AbstractPlayer, state, turn_number)
  actions, π = think(player, state)
  τ = player_temperature(player, turn_number)
  π = apply_temperature(π, τ)
  return actions[Util.rand_categorical(π)]
end

#####
##### Random Player
#####

"""
    RandomPlayer{Game} <: AbstractPlayer{Game}

A player that picks actions uniformly at random.
"""
struct RandomPlayer{Game} <: AbstractPlayer{Game} end

function think(player::RandomPlayer, state)
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

function think(p::EpsilonGreedyPlayer, state)
  actions, π = think(p.player, state)
  n = length(actions)
  η = ones(n) ./ n
  return actions, (1 - p.ϵ) * π + p.ϵ * η
end

function reset!(p::EpsilonGreedyPlayer)
  reset!(p.player)
end

function player_temperature(p::EpsilonGreedyPlayer, turn)
  return player_temperature(p.player, turn)
end

#####
##### Player with a custom temperature
#####

"""
    PlayerWithTemperature{Game, Player} <: AbstractPlayer{Game}

A wrapper on a player that enables overwriting the temperature schedule.
"""
struct PlayerWithTemperature{G, P} <: AbstractPlayer{G}
  player :: P
  temperature :: AbstractSchedule{Float64}
  function PlayerWithTemperature(p::AbstractPlayer{G}, τ) where G
    return new{G, typeof(p)}(p, τ)
  end
end

function think(p::PlayerWithTemperature, state)
  return think(p.player, state)
end

function reset!(p::PlayerWithTemperature)
  reset!(p.player)
end

function player_temperature(p::PlayerWithTemperature, turn)
  return p.temperature[turn]
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
[`AbstractSchedule`](@ref).

    MctsPlayer(oracle::MCTS.Oracle, params::MctsParams; timeout=nothing)

Construct an MCTS player from an oracle and an [`MctsParams`](@ref) structure.
If the oracle is a network, this constructor handles copying it, putting it
in test mode and copying it on the GPU (if necessary).
"""
struct MctsPlayer{G, M} <: AbstractPlayer{G}
  mcts :: M
  niters :: Int
  timeout :: Union{Float64, Nothing}
  τ :: AbstractSchedule{Float64} # Temperature
  function MctsPlayer(mcts::MCTS.Env{G}; τ, niters, timeout=nothing) where G
    @assert niters > 0
    @assert isnothing(timeout) || timeout > 0
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
    noise_α=params.dirichlet_noise_α,
    prior_temperature=params.prior_temperature)
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

function think(p::MctsPlayer, state)
  if isnothing(p.timeout) # Fixed number of MCTS simulations
    MCTS.explore!(p.mcts, state, p.niters)
  else # Run simulations until timeout
    start = time()
    while time() - start < p.timeout
      MCTS.explore!(p.mcts, state, p.niters)
    end
  end
  return MCTS.policy(p.mcts, state)
end

function player_temperature(p::MctsPlayer, turn)
  return p.τ[turn]
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

function think(p::NetworkPlayer, state)
  actions = GI.available_actions(state)
  board = GI.canonical_board(state)
  π, _ = MCTS.evaluate(p.network, board)
  return actions, π
end

#####
##### Merging different policies into a single player
#####

struct MixedPlayer{G, W, B} <: AbstractPlayer{G}
  white :: W
  black :: B
  function MixedPlayer(white, black)
    G = GameType(white)
    @assert G == GameType(black)
    return new{G, typeof(white), typeof(black)}(white, black)
  end
end

function think(p::MixedPlayer, game)
  if GI.white_playing(game)
    return think(p.white, game)
  else
    return think(p.black, game)
  end
end

function select_move(p::MixedPlayer, game, turn)
  if GI.white_playing(game)
    return select_move(p.white, game, turn)
  else
    return select_move(p.black, game, turn)
  end
end

function reset!(p::MixedPlayer)
  reset!(p.white)
  reset!(p.black)
end

function player_temperature(p::MixedPlayer, turn)
  return player_temperature(p.player, turn)
end

#####
##### Players can play against each other
#####

"""
    play_game(white, black, memory=nothing) :: Trace

Play a game between using an [`AbstractPlayer`](@ref) and return the reward
obtained by `white`.

- If the `flip_probability` argument is set to ``p``, the board
  is _flipped_ randomly at every turn with probability ``p``,
  using [`GI.apply_random_symmetry`](@ref).
"""
function play_game(player::AbstractPlayer{Game}, flip_probability=0.) where Game
  game = Game()
  trace = Trace(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      game = GI.apply_random_symmetry(game)
    end
    actions, π_target = think(player, game)
    τ = player_temperature(player, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, GI.current_state(game), GI.white_reward(game))
  end
end

function play_game(
    white::AbstractPlayer, black::AbstractPlayer, flip_probability=0.)
  return play_game(MixedPlayer(white, black), flip_probability)
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
# passed as the `memory` argument of `play_game` to record states.
# Currently, this is only used to compute redundancy statistics
struct Recorder{Game, State}
  states :: Vector{State}
  function Recorder{G}() where G
    S = GI.State(G)
    return new{G, S}([])
  end
end

push_game!(r::Recorder, wr, gl) = nothing
push_sample!(r::Recorder, s, π, wp, t) = push!(r.states, s)

function compute_redundancy(rec::Recorder{Game}) where Game
  # TODO: Excluding the inial state from redundancy statistics
  # only makes sense for deterministic games.
  init_state = GI.current_state(Game())
  noninit = filter(!=(init_state), rec.states)
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

function select_move(::Human, game, turn)
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
function interactive!(game, player)
  try
  GI.render(game)
  turn = 0
  while !GI.game_terminated(game)
    action = select_move(player, game, turn)
    GI.play!(game, action)
    GI.render(game)
    turn += 1
  end
  catch e
    isa(e, Quit) || rethrow(e)
    return
  end
end

interactive!(game, white, black) = interactive!(game, MixedPlayer(white, black))

interactive!(game::G) where G = interactive!(game, Human{G}(), Human{G}())
