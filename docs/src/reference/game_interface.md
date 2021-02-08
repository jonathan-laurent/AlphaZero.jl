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

## Interface to implement

### Game environments and game specifications

```@docs
AbstractGameSpec
AbstractGameEnv
init
spec
```
### Queries on game specs

```@docs
two_players
actions
vectorize_state
```

### Operations on environments

```@docs
set_state!
current_state
game_terminated
white_playing
actions_mask
play!
white_reward
heuristic_value
```

### Symmetries

```@docs
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

### Operations on Spec

```@docs
state_type
state_dim
state_memsize
action_type
num_actions
init(::AbstractGameSpec, state)
```

### Operations on Envs

```@docs
clone
available_actions
apply_random_symmetry!
```

## Wrapper for CommonRLInterface.jl

```@meta
CurrentModule = AlphaZero
```

```@docs
CommonRLInterfaceWrapper
```