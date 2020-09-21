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
"""
abstract type AbstractGame end

"""
    clone(::AbstractGame)

Return an independent copy of the given environment.
"""
function clone end

"""
    reset!(::AbstractGame)

Reset the game environment to an initial state.

This function is allowed to be nondeterministic for stochastic environments.

    reset!(::AbstractGame, state)

Reset the environment to a given state.
"""
function reset! end

"""
    two_players(::AbstractGame) :: Bool

Return whether or not a game is a two-players game.
"""
function two_players end

"""
    state_type(::AbstractGame)

Return the state type associated to a game environment.

State objects must be persistent or appear as such as they are stored into
the MCTS tree without copying. They also have to be comparable and hashable.
"""
function state_type(env::AbstractGame)
  return typeof(current_state(env))
end

"""
    action_type(::AbstractGame)

Return the action type associated to a game environment.
"""
function action_type(env::AbstractGame)
  return eltype(actions(env))
end

"""
    actions(::AbstractGame)

Return the vector of all game actions.
"""
function actions end

#####
##### Game functions
#####

"""
    white_playing(::AbstractGame, state) :: Bool

Return `true` if white is to play and `false` otherwise. For a one-player
game, it must always return `true`.
"""
function white_playing end

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
    symmetries(::AbstractGame, state)

Return the vector of all pairs `(s, σ)` where:
  - `s` is the image of `state` by a nonidentical symmetry
  - `σ` is the associated actions permutation, as an integer vector of
     size `num_actions(env)`.

A default implementation is provided that returns an empty vector.

Note that the urrent state of the passed environment is ignored by this function.

# Example

In the game of tic-tac-toe, there are eight symmetries that can be
obtained by composing reflexions and rotations of the board (including the
identity symmetry).
"""
function symmetries(::AbstractGame, state)
  return Tuple{typeof(state), Vector{Int}}[]
end

function test_symmetry(env, (symstate, aperm))
  mask = actions_mask(env)
  symmask = actions_mask(new_env(env, symstate))
  v = falses(length(symmask))
  v[mask] .= true
  v = v[aperm]
  return all(v[symmask]) && !any(v[.~symmask])
end

#####
##### Machine learning interface
#####

"""
    vectorize_state(env::AbstractGame, state) :: Array{Float32}

Return a vectorized representation of a given state.

Note that the current state of the passed environment is not considered.
"""
function vectorize_state end

#####
##### Interface for interactive exploratory tools
#####

"""
    action_string(::AbstractGame, action) :: String

Return a human-readable string representing the provided action.
"""
function action_string end

"""
    parse_action(::AbstractGame, str::String)

Return the action described by string `str` or `nothing`
if `str` does not denote a valid action.
"""
function parse_action end

"""
    read_state(env::AbstractGame)

Read a state from the standard input.
Return the corresponding state (with type `state_type(env)`)
or `nothing` in case of an invalid input.
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
    new_env(env::AbstractGame)

Return a fresh copy of the current environment, reset to an initial state.
"""
function new_env(env::AbstractGame)
  env = clone(env)
  reset!(env)
  return env
end

"""
    new_env(env::AbstractGame, state)

Return a fresh copy of the current environment, reset to a given state.
"""
function new_env(env::AbstractGame, state)
  env = clone(env)
  reset!(env, state)
  return env
end

"""
    num_actions(::AbstractGame)

Return the total number of actions associated with a game.
"""
num_actions(::Type{G}) where G = length(actions(G))

"""
    available_actions(::AbstractGame)

Return the vector of all available actions.
"""
function available_actions(env::AbstractGame)
  Game = typeof(env)
  mask = actions_mask(env)
  return actions(Game)[mask]
end

"""
    state_dim(::AbstractGame)

Return a tuple that indicates the shape of a vectorized state representation.
"""
state_dim(env) = size(vectorize_state(env, current_state(env)))

"""
    apply_random_symmetry!(::AbstractGame)

Apply a random symmetry to the current game state (see [`symmetries`](@ref)).
"""
function apply_random_symmetry!(env)
  symstate, _ = rand(symmetries(env))
  reset!(env, symstate)
  return
end

function state_memsize(env::AbstractGame)
  return Base.summarysize(current_state(env))
end

end