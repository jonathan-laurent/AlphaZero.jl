"""
    GameInterface

A generic interface for zero-sum, symmetric games.

# Assumptions

The `Action` type must be "symmetric" in the following sense
```julia
available_actions(s) ==
  available_actions(State(board_symmetric(s), !white_playing(s)))
```
"""
module GameInterface

#####
##### API functions
#####

# Types
function Board end
function Action end

# Constructors
# - Game()
# - Game(board, white_playing=true)
# - Base.copy(::Game)

# Game functions
function white_playing end
function white_reward end
function board end
function board_symmetric end
function available_actions end
function play! end

# Machine learning interface
function board_dim end
function vectorize_board end
function num_actions end
function action end
function action_id end

# Used by the game interface and the exploration tools
function action_string end
function parse_action end
function read_state end
function print_state end


#####
##### Derived functions
#####

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
##### Minimalistic game interface
#####

abstract type Player end

struct Human <: Player end

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

function interactive!(game, white::Player, black::Player)
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

end
