"""
A generic interface for two-players, symmetric, zero-sum games.
"""
module GameInterface

export AbstractPlayer

import ..Util
using ..Util: @unimplemented

#####
##### Types
#####

"""
    AbstractGame

Abstract base type for a game state.

# Constructors

Any subtype `Game` must implement `Base.copy` along with
the following constructors:

    Game()

Return the initial state of the game.

    Game(board, white_playing=true)

Return the unique state specified by a board and a current player.
By convention, the first player to play is called _white_ and
the other is called _black_.
"""
abstract type AbstractGame end

"""
    Board(Game::Type{<:AbstractGame})

Return the board type corresponding to `Game`.

Board objects must be persistent or appear as such as they are stored into
the MCTS tree without copying.

# Remark

A game state (of type [`AbstractGame`](@ref)) is characterized by two pieces
of information: the board state and the identity of the player to play next.
There are two reasons for having a separate `Board` type:

  - This separation allows the `Game` object to store redundant state
    information, typically for caching expensive computations.
  - This separation enables leveraging the symmetry between players by
    storing every board in the MCTS tree from the perspective of the current
    player (as if white were to play next).
"""
function Board(::Type{<:AbstractGame})
  @unimplemented
end

"""
    Action(Game::Type{<:AbstractGame})

Return the action type corresponding to `Game`.

Actions must be "symmetric" in the following sense:

```
available_actions(s) ==
  available_actions(Game(board_symmetric(s), !white_playing(s)))
```
"""
function Action(::Type{<:AbstractGame})
  @unimplemented
end

"""
    actions(::Type{<:AbstractGame})

Return the vector of all game actions.
"""
function actions(::Type{<:AbstractGame})
  @unimplemented
end

function Base.copy(::AbstractGame)
  @unimplemented
end

#####
##### Game functions
#####

"""
    white_playing(state::AbstractGame) :: Bool

Return `true` if white is to play and `false` otherwise.
"""
function white_playing(::AbstractGame)
  @unimplemented
end

"""
    white_reward(state::AbstractGame)

Return `nothing` if the game hasn't ended. Otherwise, return a
reward for the white player as a number between -1 and 1.
"""
function white_reward(::AbstractGame)
  @unimplemented
end

"""
    board(state::AbstractGame)

Return the game board.
"""
function board(::AbstractGame)
  @unimplemented
end

"""
    board_symmetric(state::AbstractGame)

Return the symmetric of the game board
(where the roles of black and white are swapped).
"""
function board_symmetric(::AbstractGame)
  @unimplemented
end

"""
    actions_mask(state::AbstractGame)

Return a boolean mask indicating what actions are available from `state`.

The following identities must hold:

    game_terminated(state) || any(actions_mask(state))
    length(actions_mask(state)) == length(actions(typeof(state)))
"""
function actions_mask(state::AbstractGame)
  @unimplemented
end

"""
    play!(state::AbstractGame, action)

Update the game state by making the current player perform `action`.
"""
function play!(state::AbstractGame, action)
  @unimplemented
end

"""
    heuristic_value(state::AbstractGame)

Return a heuristic estimate of the state value for the current player.

The given state must be nonfinal and returned values must belong to the
``(-∞, ∞)`` interval. Also, implementations of this function must be
antisymmetric in the sense that:
```
heuristic_value(s) ==
  - heuristic_value(Game(board_symmetric(s), white_playing(s)))
```

This function is not needed by AlphaZero but it is useful for building
baselines such as minmax players.
"""
function heuristic_value(state::AbstractGame)
  @unimplemented
end

#####
##### Symmetries
#####

"""
    symmetries(::Type{G}, board) where {G <: AbstractGame}

Return the vector of all pairs `(b, σ)` where:
  - `b` is the image of `board` by a nonidentical symmetry
  - `σ` is the associated actions permutation, as an integer vector of
     size `num_actions(Game)`.

A default implementation is provided that returns an empty vector.
"""
function symmetries(::Type{G}, board) where {G <: AbstractGame}
  return Tuple{Board(G), Vector{Int}}[]
end

function test_symmetry(Game, board, (symboard, aperm))
  syms = symmetries
  mask = actions_mask(Game(board))
  symmask = actions_mask(Game(symboard))
  v = falses(length(symmask))
  v[mask] .= true
  v = v[aperm]
  return all(v[symmask]) && !any(v[.~symmask])
end

#####
##### Machine learning interface
#####

"""
    vectorize_board(::Type{<:AbstractGame}, board) :: Array{Float32}

Return a vectorized representation of a board.
"""
function vectorize_board(::Type{<:AbstractGame}, board)
  @unimplemented
end

#####
##### Interface for interactive exploratory tools
#####

"""
    action_string(::Type{<:AbstractGame}, action) :: String

Return a human-readable string representing the provided action.
"""
function action_string(::Type{<:AbstractGame}, action)
  @unimplemented
end

"""
    parse_action(::Type{<:AbstractGame}, str::String)

Return the action described by string `str` or `nothing`
if `str` does not denote a valid action.
"""
function parse_action(::Type{<:AbstractGame}, ::String)
  @unimplemented
end

"""
    read_state(::Type{G}) where G <: AbstractGame :: Union{G, Nothing}

Read a state description from the standard input.
Return the corresponding state or `nothing` in case of an invalid input.
"""
function read_state(::Type{<:AbstractGame})
  @unimplemented
end

"""
    print_state(state::AbstractGame)

Print a state on the standard output.
"""
function print_state(::AbstractGame)
  @unimplemented
end

#####
##### Derived functions
#####

"""
    game_terminated(::AbstractGame)

Return a boolean indicating whether or not a game is in a terminal state.
"""
game_terminated(state) = !isnothing(white_reward(state))

"""
    num_actions(::Type{G})

Return the total number of actions associated with a game.
"""
num_actions(::Type{G}) where G = length(actions(G))

"""
    available_actions(state::AbstractGame)

Return the vector of all available actions in a given state.
"""
function available_actions(state::AbstractGame)
  Game = typeof(state)
  mask = actions_mask(state)
  return actions(Game)[mask]
end

"""
    board_dim(::Type{G})

Return a tuple that indicates the shape of a vectorized board representation.
"""
board_dim(::Type{G}) where G = size(vectorize_board(G, board(G())))

function canonical_board(state)
  return white_playing(state) ? board(state) : board_symmetric(state)
end

"""
    random_symmetric_state(::AbstractGame)

Return a fresh new state that is the image of the given state by a random
symmetry (see [`symmetries`](@ref)).
"""
function random_symmetric_state(state::Game) where {Game <: AbstractGame}
  bsym, _ = rand(symmetries(Game, board(state)))
  return Game(bsym, white_playing(state))
end

function board_memsize(::Type{G}) where G
  return Base.summarysize(board(G()))
end

symmetric_reward(r::Real) = -r

end

"""
    GameType(T)

Return the `Game` type associated with an object
(such as a network, a player, an MCTS environment...)
"""
function GameType(T)
  @unimplemented
end
