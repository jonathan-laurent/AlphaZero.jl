# [Game Interface](@id game_interface)

```@meta
CurrentModule = AlphaZero.GameInterface
```

```@docs
GameInterface
```

## Types

```@docs
AbstractGame
Board
Action
```

## Game Functions

```@docs
white_playing
white_reward
board
board_symmetric
actions
actions_mask
available_actions
play!
heuristic_value
vectorize_board
symmetries
```

## Interface for Interactive Tools

```@docs
action_string
parse_action
read_state
print_state
```

```@meta
CurrentModule = AlphaZero
```

## Utilities

```@docs
GameType
```
