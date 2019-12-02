# MCTS

A generic, standalone implementation of asynchronous Monte Carlo Tree Search.
It can be used on any game that implements the [`GameInterface`](@ref
Main.AlphaZero.GameInterface) interface and with any external oracle.

```@meta
CurrentModule = AlphaZero.MCTS
```

## Oracles

```@docs
Oracle
evaluate(::Oracle, board, actions)
evaluate_batch(::Oracle, batch)
RolloutOracle
```

## Environment

```@docs
Env
explore!
policy
reset!
```

## Profiling utilities

```@docs
inference_time_ratio
memory_footprint_per_node
approximate_memory_footprint
average_exploration_depth
```
