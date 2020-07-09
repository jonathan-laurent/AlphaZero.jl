"""
A generic interface for two-players, zero-sum games.

Stochastic games and intermediate rewards are supported. By convention,
rewards are expressed from the point of view of the player called _white_.
"""
module GameInterface

export AbstractGame

import ..Util

#####
##### Types
#####

"""
    AbstractGame

Abstract base type for a game environment.

# Constructors

Any subtype `Game` must implement the following constructors:

    Game()

Return an initialized game environment. Note that this constructor does not
have to be deterministic.

    Game(state)

Return a fresh game environment starting at a given state.
"""
abstract type AbstractGame end

function Base.copy(env::Game) where {Game <: AbstractGame}
  return Game(current_state(env))
end

"""
    two_players(::Type{<:AbstractGame}) :: Bool

Return whether or not a game is a two-players game.
"""
function two_players end

"""
    State(Game::Type{<:AbstractGame})

Return the state type corresponding to `Game`.

State objects must be persistent or appear as such as they are stored into
the MCTS tree without copying. They also have to be comparable and hashable.
"""
function State end

"""
    Action(Game::Type{<:AbstractGame})

Return the action type corresponding to `Game`.
"""
function Action end

"""
    actions(::Type{<:AbstractGame})

Return the vector of all game actions.
"""
function actions end

#####
##### Game functions
#####

"""
    white_playing(::Type{<:AbstractGame}, state) :: Bool
    white_playing(env::AbstractGame)
      = white_playing(typeof(env), current_state(env))

Return `true` if white is to play and `false` otherwise. For a one-player
game, it must always return `true`.
"""
function white_playing end

function white_playing(env::AbstractGame)
  return white_playing(typeof(env), current_state(env))
end

"""
    white_reward(env::AbstractGame)

Return the intermediate reward obtained by the white player after the last
transition step. The result is undetermined when called at an initial state.
"""
function white_reward end

"""
    current_state(env::AbstractGame)

Return the game state (which is persistent).
"""
function current_state end

"""
    game_terminated(::AbstractGame)

Return a boolean indicating whether or not the game is in a terminal state.
"""
function game_terminated end

"""
    actions_mask(env::AbstractGame)

Return a boolean mask indicating what actions are available from `env`.

The following identities must hold:

  - `game_terminated(env) || any(actions_mask(env))`
  - `length(actions_mask(env)) == length(actions(typeof(env)))`
"""
function actions_mask end

"""
    play!(env::AbstractGame, action)

Update the game environment by making the current player perform `action`.
Note that this function does not have to be deterministic.
"""
function play! end

"""
    heuristic_value(env::AbstractGame)

Return a heuristic estimate of the state value for the current player.

The given state must be nonfinal and returned values must belong to the
``(-∞, ∞)`` interval.

This function is not needed by AlphaZero but it is useful for building
baselines such as minmax players.
"""
function heuristic_value end

#####
##### Symmetries
#####

"""
    symmetries(::Type{G}, state) where {G <: AbstractGame}

Return the vector of all pairs `(s, σ)` where:
  - `s` is the image of `state` by a nonidentical symmetry
  - `σ` is the associated actions permutation, as an integer vector of
     size `num_actions(Game)`.

A default implementation is provided that returns an empty vector.

# Example

In the game of tic-tac-toe, there are eight symmetries that can be
obtained by composing reflexions and rotations of the board (including the
identity symmetry).
"""
function symmetries(::Type{G}, state) where {G <: AbstractGame}
  return Tuple{State(G), Vector{Int}}[]
end

function test_symmetry(Game, state, (symstate, aperm))
  syms = symmetries
  mask = actions_mask(Game(state))
  symmask = actions_mask(Game(symstate))
  v = falses(length(symmask))
  v[mask] .= true
  v = v[aperm]
  return all(v[symmask]) && !any(v[.~symmask])
end

#####
##### Machine learning interface
#####

"""
    vectorize_state(::Type{<:AbstractGame}, state) :: Array{Float32}

Return a vectorized representation of a state.
"""
function vectorize_state end

#####
##### Interface for interactive exploratory tools
#####

"""
    action_string(::Type{<:AbstractGame}, action) :: String

Return a human-readable string representing the provided action.
"""
function action_string end

"""
    parse_action(::Type{<:AbstractGame}, str::String)

Return the action described by string `str` or `nothing`
if `str` does not denote a valid action.
"""
function parse_action end

"""
    read_state(::Type{G}) where G <: AbstractGame :: Union{State(G), Nothing}

Read a state from the standard input.
Return the corresponding state or `nothing` in case of an invalid input.
"""
function read_state end

"""
    render(env::AbstractGame)

Print the game state on the standard output.
"""
function render end

#####
##### Derived functions
#####

"""
    num_actions(::Type{G})

Return the total number of actions associated with a game.
"""
num_actions(::Type{G}) where G = length(actions(G))

"""
    available_actions(env::AbstractGame)

Return the vector of all available actions.
"""
function available_actions(env::AbstractGame)
  Game = typeof(env)
  mask = actions_mask(env)
  return actions(Game)[mask]
end

"""
    state_dim(::Type{G})

Return a tuple that indicates the shape of a vectorized state representation.
"""
state_dim(::Type{G}) where G = size(vectorize_state(G, current_state(G())))

"""
    apply_random_symmetry(::AbstractGame)

Return a fresh new state that is the image of the given state by a random
symmetry (see [`symmetries`](@ref)).
"""
function apply_random_symmetry(env::Game) where {Game <: AbstractGame}
  symstate, _ = rand(symmetries(Game, current_state(env)))
  return Game(symstate)
end

function state_memsize(::Type{G}) where G
  return Base.summarysize(current_state(G()))
end

end

"""
    GameType(T)

Return the `Game` type associated with an object
(such as a network, a player, an MCTS environment...)
"""
function GameType end
