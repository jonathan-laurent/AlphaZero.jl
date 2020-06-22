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
    think(::AbstractPlayer, game)

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
    player_temperature(::AbstractPlayer, game, turn_number)

Return the player temperature, given the number of actions that have
been played before by both players in the current game.

A default implementation is provided that always returns 1.
"""
function player_temperature(::AbstractPlayer, game, turn)
  return 1.0
end

"""
    select_move(player::AbstractPlayer, game, turn_number)

Return a single action. A default implementation is provided that samples
an action according to the distribution computed by [`think`](@ref), with a
temperature given by [`player_temperature`](@ref).
"""
function select_move(player::AbstractPlayer, game, turn_number)
  actions, π = think(player, game)
  τ = player_temperature(player, game, turn_number)
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

function think(player::RandomPlayer, game)
  actions = GI.available_actions(game)
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

function think(p::EpsilonGreedyPlayer, game)
  actions, π = think(p.player, game)
  n = length(actions)
  η = ones(n) ./ n
  return actions, (1 - p.ϵ) * π + p.ϵ * η
end

function reset!(p::EpsilonGreedyPlayer)
  reset!(p.player)
end

function player_temperature(p::EpsilonGreedyPlayer, game, turn)
  return player_temperature(p.player, game, turn)
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

function think(p::PlayerWithTemperature, game)
  return think(p.player, game)
end

function reset!(p::PlayerWithTemperature)
  reset!(p.player)
end

function player_temperature(p::PlayerWithTemperature, game, turn)
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
    gamma=params.gamma,
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
    gamma=params.gamma,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α)
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature)
end

function think(p::MctsPlayer, game)
  if isnothing(p.timeout) # Fixed number of MCTS simulations
    MCTS.explore!(p.mcts, game, p.niters)
  else # Run simulations until timeout
    start = time()
    while time() - start < p.timeout
      MCTS.explore!(p.mcts, game, p.niters)
    end
  end
  return MCTS.policy(p.mcts, game)
end

function player_temperature(p::MctsPlayer, game, turn)
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
  function NetworkPlayer(nn::AbstractNetwork{G}; use_gpu=false) where G
    nn = Network.copy(nn, on_gpu=use_gpu, test_mode=true)
    return new{G, typeof(nn)}(nn)
  end
end

function think(p::NetworkPlayer, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  π, _ = MCTS.evaluate(p.network, state)
  return actions, π
end

#####
##### Merging two players into one
#####

struct TwoPlayers{G, W, B} <: AbstractPlayer{G}
  white :: W
  black :: B
  function TwoPlayers(white, black)
    G = GameType(white)
    @assert G == GameType(black)
    return new{G, typeof(white), typeof(black)}(white, black)
  end
end

function think(p::TwoPlayers, game)
  if GI.white_playing(game)
    return think(p.white, game)
  else
    return think(p.black, game)
  end
end

function select_move(p::TwoPlayers, game, turn)
  if GI.white_playing(game)
    return select_move(p.white, game, turn)
  else
    return select_move(p.black, game, turn)
  end
end

function reset!(p::TwoPlayers)
  reset!(p.white)
  reset!(p.black)
end

function player_temperature(p::TwoPlayers, game, turn)
  if GI.white_playing(game)
    return player_temperature(p.white, game, turn)
  else
    return player_temperature(p.black, game, turn)
  end
end

#####
##### Players can play against each other
#####

"""
    play_game(player; flip_probability=0.) :: Trace

Play a game between using an [`AbstractPlayer`](@ref) and return the reward
obtained by `white`.

- If the `flip_probability` argument is set to ``p``, the board
  is _flipped_ randomly at every turn with probability ``p``,
  using [`GI.apply_random_symmetry`](@ref).
"""
function play_game(player; flip_probability=0.)
  Game = GameType(player)
  game = Game()
  trace = Trace{Game}(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      game = GI.apply_random_symmetry(game)
    end
    actions, π_target = think(player, game)
    τ = player_temperature(player, game, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, GI.current_state(game), π_target, GI.white_reward(game))
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

Note that this function can only be used with two-player games.

# Arguments

  - `handler`: this function is called after each simulated
     game with three arguments: the game number `i`, the reward `r` for the
     contender player and the trace `t`
  - `ngames`: number of games to play

# Optional keyword arguments
  - `reset_every`: if set, players are reset every `reset_every` games
  - `color_policy`: determines the [`ColorPolicy`](@ref),
     which is `ALTERNATE_COLORS` by default
  - `memory=nothing`: memory to use to record samples
  - `flip_probability=0.`: see [`play_game`](@ref)
"""
function pit(
    handler, contender, baseline, num_games; gamma,
    reset_every=nothing, color_policy=ALTERNATE_COLORS, flip_probability=0.)
  Game = GameType(contender)
  @assert GI.two_players(Game)
  baseline_white = (color_policy != CONTENDER_WHITE)
  zsum = 0.
  for i in 1:num_games
    white = baseline_white ? baseline : contender
    black = baseline_white ? contender : baseline
    trace =
      play_game(TwoPlayers(white, black), flip_probability=flip_probability)
    z = total_reward(trace, gamma)
    baseline_white && (z = -z)
    zsum += z
    handler(i, z, trace)
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

function compute_redundancy(Game, states)
  # TODO: Excluding the inial state from redundancy statistics
  # only makes sense for deterministic games.
  init_state = GI.current_state(Game())
  noninit = filter(!=(init_state), states)
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
