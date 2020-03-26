# [Game Interface](@id game_interface)

```@meta
CurrentModule = AlphaZero
```

```@docs
GameInterface
```

```@meta
CurrentModule = AlphaZero.GameInterface
```

## Mandatory Interface

### Types

```@docs
AbstractGame
Board
Action
```

### Game Functions

```@docs
white_playing
white_reward
board
board_symmetric
actions
actions_mask
play!
heuristic_value
vectorize_board
symmetries
```

### Interface for Interactive Tools

```@docs
action_string
parse_action
read_state
print_state
```

## Derived Functions

```@docs
state_symmetric
game_terminated
num_actions
available_actions
board_dim
random_symmetric_state
```
