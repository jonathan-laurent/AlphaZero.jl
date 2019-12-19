"""
A generic interface for zero-sum, symmetric games.
"""
module GameInterface

export AbstractPlayer

using ..Util: @unimplemented, rand_categorical

#####
##### Types
#####

"""
    AbstractGame

Abstract base type for a game state.

# Constructors

Any subtype `Game` must implement the following constructors:

    Game()

Return the initial state of the game.

    Game(board, white_playing=true)

Return the unique state specified by a board and a current player.
"""
abstract type AbstractGame end

"""
    Board(Game::Type{<:AbstractGame})

Return the board type corresponding to `Game`.

Board objects must be persistent or appear as such.
"""
function Board(::Type{<:AbstractGame})
  @unimplemented
end

"""
    Action(Game::Type{<:AbstractGame})

Return the action type corresponding to `Game`.

Actions must be "symmetric" in the following sense:

```julia
available_actions(s) ==
  available_actions(Game(board_symmetric(s), !white_playing(s)))
```
"""
function Action(::Type{<:AbstractGame})
  @unimplemented
end

"""
    Base.copy(::AbstractGame)

Return a fresh copy of a game state.
"""
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
    available_actions(state::AbstractGame)

Return the vector of all available actions, which must be nonempty if
`isnothing(white_reward(state))`.
"""
function available_actions(::AbstractGame)
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
##### Machine learning interface
#####

"""
    vectorize_board(::Type{<:AbstractGame}, board) :: Vector{Float32}

Return a vectorized representation of a board.
"""
function vectorize_board(::Type{<:AbstractGame}, board)
  @unimplemented
end

"""
    num_actions(::Type{<:AbstractGame}) :: Int

Return the total number of actions for a game.
"""
function num_actions(::Type{<:AbstractGame})
  @unimplemented
end

"""
    action_id(G::Type{<:AbstractGame}, action) :: Int

Map each action to a unique number in the range `1:num_actions(G)`.
"""
function action_id(::Type{<:AbstractGame}, action)
  @unimplemented
end

"""
    action(::Type{<:AbstractGame}, Int)

Inverse function of [`action_id`](@ref GameInterface.action_id).

Map an action identifier to an actual action.
"""
function action(::Type{<:AbstractGame}, id)
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

board_dim(::Type{G}) where G = size(vectorize_board(G, board(G())))

function actions_mask(::Type{G}, available_actions) where G
  nactions = num_actions(G)
  mask = falses(nactions)
  for a in available_actions
    mask[action_id(G, a)] = true
  end
  return mask
end

function canonical_board(state)
  white_playing(state) ? board(state) : board_symmetric(state)
end

function board_memsize(::Type{G}) where G
  return Base.summarysize(board(G()))
end

#####
##### Abstract type for players
#####

abstract type AbstractPlayer{Game} end

"""
   think(::AbstractPlayer, state, turn=nothing)

Return a probability distribution over actions as a `(actions, π)` pair.
"""
function think(::AbstractPlayer, state, turn=nothing)
  @unimplemented
end

"""
    reset_player!(::AbstractPlayer)

Reset the internal memory of a player. The default implementation does nothing.
"""
function reset_player!(::AbstractPlayer)
  return
end

function select_move(player::AbstractPlayer, state, turn=nothing)
  actions, π = think(player, state, turn)
  #println(ceil.(100 * π), "\n")
  return actions[Util.rand_categorical(π)]
end

#####
##### Random Player
#####

struct RandomPlayer{Game} <: AbstractPlayer{Game} end

function think(player::RandomPlayer, state, turn)
  actions = GI.available_actions(state)
  n = length(actions)
  π = ones(n) ./ length(actions)
  return π
end

#####
##### Minimalistic interface for humans
#####

struct Human{Game} <: AbstractPlayer{Game} end

struct Quit <: Exception end

function select_move(::Human, game)
  a = nothing
  while isnothing(a) || a ∉ available_actions(game)
    print("> ")
    str = readline()
    print("\n")
    isempty(str) && throw(Quit())
    a = parse_action(game, str)
  end
  return a
end

"""
    interactive!(game, white, black)

Launch an interactive session for `game` between players `white` and `black`.
Both players have type `AbstractPlayer` and one of them is typically `Human`.
"""
function interactive!(game, white, black)
  try
  print_state(game)
  while isnothing(white_reward(game))
    player = white_playing(game) ? white : black
    action = select_move(player, game)
    play!(game, action)
    print_state(game)
  end
  catch e
    isa(e, Quit) || rethrow(e)
    return
  end
end

interactive!(game::G) where G = interactive!(game, Human{G}(), Human{G}())

end
