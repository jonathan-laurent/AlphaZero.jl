# [Players](@id player)

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
MctsPlayer
RandomPlayer
NetworkPlayer
EpsilonGreedyPlayer
PlayerWithTemperature
TwoPlayers
```

## Derived Functions

```@docs
play_game
Trace
Base.push!(::Trace, π, r, s)
interactive!
Human
```

### Distributed Simulator

```@docs
Simulator
record_trace
ColorPolicy
simulate
simulate_distributed
```
