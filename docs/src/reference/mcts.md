# MCTS

```@meta
CurrentModule = AlphaZero.MCTS
```

```@docs
MCTS
```

## Oracles

```@docs
Oracle
evaluate(::Oracle, board, actions)
evaluate_batch(::Oracle, batches)
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
