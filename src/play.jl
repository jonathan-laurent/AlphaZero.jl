#####
##### Interface for players
#####

"""
    AbstractPlayer

Abstract type for a game player.
"""
abstract type AbstractPlayer end

"""
    think(::AbstractPlayer, game)

Return a probability distribution over available actions as a `(actions, π)` pair.
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
    RandomPlayer <: AbstractPlayer

A player that picks actions uniformly at random.
"""
struct RandomPlayer <: AbstractPlayer end

function think(::RandomPlayer, game)
  actions = GI.available_actions(game)
  n = length(actions)
  π = ones(n) ./ length(actions)
  return actions, π
end

#####
##### Epsilon-greedy player
#####

"""
    EpsilonGreedyPlayer{Player} <: AbstractPlayer

A wrapper on a player that makes it choose a random move
with a fixed ``ϵ`` probability.
"""
struct EpsilonGreedyPlayer{P} <: AbstractPlayer
  player :: P
  ϵ :: Float64
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
    PlayerWithTemperature{Player} <: AbstractPlayer

A wrapper on a player that enables overwriting the temperature schedule.
"""
struct PlayerWithTemperature{P} <: AbstractPlayer
  player :: P
  temperature :: AbstractSchedule{Float64}
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
    MctsPlayer{MctsEnv} <: AbstractPlayer

A player that selects actions using MCTS.

# Constructors

    MctsPlayer(mcts::MCTS.Env; τ, niters, timeout=nothing)

Construct a player from an MCTS environment. When computing each move:

- if `timeout` is provided,
  MCTS simulations are executed for `timeout` seconds by groups of `niters`
- otherwise, `niters` MCTS simulations are run

The temperature parameter `τ` can be either a real number or a
[`AbstractSchedule`](@ref).

    MctsPlayer(game_spec::AbstractGameSpec, oracle,
               params::MctsParams; timeout=nothing)

Construct an MCTS player from an oracle and an [`MctsParams`](@ref) structure.
"""
struct MctsPlayer{M} <: AbstractPlayer
  mcts :: M
  niters :: Int
  timeout :: Union{Float64, Nothing}
  τ :: AbstractSchedule{Float64} # Temperature
  function MctsPlayer(mcts::MCTS.Env; τ, niters, timeout=nothing)
    @assert niters > 0
    @assert isnothing(timeout) || timeout > 0
    new{typeof(mcts)}(mcts, niters, timeout, τ)
  end
end

# Alternative constructor
function MctsPlayer(
    game_spec::AbstractGameSpec, oracle, params::MctsParams; timeout=nothing)
  mcts = MCTS.Env(game_spec, oracle,
    gamma=params.gamma,
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
function RandomMctsPlayer(game_spec::AbstractGameSpec, params::MctsParams)
  oracle = MCTS.RandomOracle()
  mcts = MCTS.Env(game_spec, oracle,
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
    NetworkPlayer{Net} <: AbstractPlayer

A player that uses the policy output by a neural network directly,
instead of relying on MCTS. The given neural network must be in test mode.
"""
struct NetworkPlayer{N} <: AbstractPlayer
  network :: N
end

function think(p::NetworkPlayer, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  π, _ = p.network(state)
  return actions, π
end

#####
##### Merging two players into one
#####

"""
    TwoPlayers <: AbstractPlayer

If `white` and `black` are two [`AbstractPlayer`](@ref), then
`TwoPlayers(white, black)` is a player that behaves as `white` when `white`
is to play and as `black` when `black` is to play.
"""
struct TwoPlayers{W, B} <: AbstractPlayer
  white :: W
  black :: B
end

flipped_colors(p::TwoPlayers) = TwoPlayers(p.black, p.white)

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
    play_game(gspec::AbstractGameSpec, player; flip_probability=0.) :: Trace

Simulate a game by an [`AbstractPlayer`](@ref).

- For two-player games, please use [`TwoPlayers`](@ref).
- If the `flip_probability` argument is set to ``p``, the board
  is _flipped_ randomly at every turn with probability ``p``,
  using [`GI.apply_random_symmetry!`](@ref).
"""
function play_game(gspec, player; flip_probability=0.)
  game = GI.init(gspec)
  trace = Trace(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      GI.apply_random_symmetry!(game)
    end
    actions, π_target = think(player, game)
    τ = player_temperature(player, game, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, π_target, GI.white_reward(game), GI.current_state(game))
  end
end

#####
##### Simple utilities to play an interactive game
#####

"""
    Human <: AbstractPlayer

Human player that queries the standard input for actions.

Does not implement [`think`](@ref) but instead implements
[`select_move`](@ref) directly.
"""
struct Human <: AbstractPlayer end

struct Quit <: Exception end

function select_move(::Human, game, turn)
  a = nothing
  while isnothing(a) || a ∉ GI.available_actions(game)
    print("> ")
    str = readline()
    print("\n")
    isempty(str) && throw(Quit())
    a = GI.parse_action(GI.spec(game), str)
  end
  return a
end

"""
    interactive!(game)
    interactive!(gspec)
    interactive!(game, player)
    interactive!(gspec, player)
    interactive!(game, white, black)
    interactive!(gspec, white, black)

Launch a possibly interactive game session.

This function takes either an `AbstractGameSpec` or `AbstractGameEnv` as an argument.
"""
function interactive! end

function interactive!(game::AbstractGameEnv, player)
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

interactive!(gspec::AbstractGameSpec, player) = interactive!(GI.init(gspec), player)

interactive!(game, white, black) = interactive!(game, TwoPlayers(white, black))

interactive!(game) = interactive!(game, Human(), Human())
