# [Game Interface](@id game_interface)

```@meta
CurrentModule = AlphaZero
```

```@docs
GameInterface
```

A test suite is provided in the `AlphaZero.Scripts` to check the compliance of your
environment with this interface.

```@meta
CurrentModule = AlphaZero.GameInterface
```

## Mandatory Interface

The game interface of AlphaZero.jl differs from many standard RL interfaces by making
a distinction between a game **specification** and a game **environment**:

  - A _specification_ holds all _static_ information about a game, which does not
    depend on the current state (e.g. the world dimensions in a grid world environment)
  - In contrast, an _environment_ holds information about the current state of the game
    (e.g. the player's position in a grid-world environment).

### Game Specifications

```@docs
AbstractGameSpec
two_players
actions
vectorize_state
```

### Game Environments

```@docs
AbstractGameEnv
init
spec
set_state!
current_state
game_terminated
white_playing
actions_mask
play!
white_reward
```

## Optional Interface

### Interface for Interactive Tools

These functions are required for the default [User Interface](@ref ui) to work well.

```@docs
action_string
parse_action
read_state
render
```

### Other Optional Functions

```@docs
heuristic_value
symmetries
```

## Derived Functions

### Operations on Specifications

```@docs
state_type
state_dim
state_memsize
action_type
num_actions
init(::AbstractGameSpec, state)
```

### Operations on Environments

```@docs
clone
available_actions
apply_random_symmetry!
```

## [Wrapper for CommonRLInterface.jl](@id common_rl_intf)

```@meta
CurrentModule = AlphaZero
```

```@docs
CommonRLInterfaceWrapper
CommonRLInterfaceWrapper.Env
CommonRLInterfaceWrapper.Spec
```