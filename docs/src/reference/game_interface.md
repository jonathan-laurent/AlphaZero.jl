# [Game Interface](@id game_interface)

```@meta
CurrentModule = AlphaZero.GameInterface
```

The [`GameInterface`](@ref Main.AlphaZero.GameInterface) module provides a
generic interface for two-players, zero-sum, symmetric board games.

  * Types, traits and constructors
    - [`Game()`](@ref AbstractGame)
    - [`Game(board)`](@ref AbstractGame)
    - [`Game(board, white_playing)`](@ref AbstractGame)
    - [`Base.copy(game)`](@ref Base.copy(::AbstractGame))
    - [`Board(Game)`](@ref Board)
    - [`Action(Game)`](@ref Action)
  * Game functions
    - [`white_playing(state)`](@ref white_playing)
    - [`white_reward(state)`](@ref white_reward)
    - [`board(state)`](@ref board)
    - [`board_symmetric(state)`](@ref board_symmetric)
    - [`available_actions(state)`](@ref available_actions)
    - [`play!(state, action)`](@ref play!)
  * Machine learning interface
    - [`vectorize_board(Game, board)`](@ref vectorize_board)
    - [`num_actions(Game)`](@ref num_actions)
    - [`action_id(Game, action)`](@ref action_id)
    - [`action(Game, action_id)`](@ref action)
  * Interface for interactive tools
    - [`action_string(Game, action)`](@ref action_string)
    - [`parse_action(Game, str)`](@ref parse_action)
    - [`read_state(Game)`](@ref read_state)
    - [`print_state(state)`](@ref print_state)

## Types

```@docs
AbstractGame
Base.copy(::AbstractGame)
Board
Action
```

## Game Functions

```@docs
white_playing
white_reward
board
board_symmetric
available_actions
play!
```

## Machine Learning Interface

```@docs
vectorize_board
num_actions
action_id
action
```

## Interface for Interactive Tools

```@docs
action_string
parse_action
read_state
print_state
```
