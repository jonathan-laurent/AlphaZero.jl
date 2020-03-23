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
```

## Player Instances

```@docs
MctsPlayer
RandomPlayer
NetworkPlayer
EpsilonGreedyPlayer
```

## Derived Functions

```@docs
play_game
pit
ColorPolicy
interactive!
Human
```
