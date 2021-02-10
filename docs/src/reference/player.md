# [Players and Simulations](@id player)

```@meta
CurrentModule = AlphaZero
```

## Player Interface

```@docs
AbstractPlayer
think
select_move
reset_player!
player_temperature
```

## Player Instances

```@docs
AlphaZeroPlayer
MctsPlayer
RandomPlayer
NetworkPlayer
EpsilonGreedyPlayer
PlayerWithTemperature
TwoPlayers
```

## Game Simulations

### Simulation traces

```@docs
Trace
Base.push!(::Trace, π, r, s)

```

### Playing a single game

```@docs
play_game
```

### Playing multiple games in a distibuted fashion

```@docs
Simulator
record_trace
simulate
simulate_distributed
```

### Utilities for playing interactive games

```@docs
Human
interactive!
```