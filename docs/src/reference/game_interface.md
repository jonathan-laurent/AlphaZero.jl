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
State
Action
two_players
```

### Game Functions

```@docs
game_terminated
white_playing
white_reward
current_state
actions
actions_mask
play!
heuristic_value
vectorize_state
symmetries
```

### Interface for Interactive Tools

```@docs
action_string
parse_action
read_state
render
```

## Derived Functions

```@docs
num_actions
available_actions
state_dim
apply_random_symmetry
```
