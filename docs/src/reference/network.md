# [Network Interface](@id network_interface)

```@meta
CurrentModule = AlphaZero.Network
```

This module defines a generic, framework-agnostic interface for neural network
oracles.

## Mandatory interface

```@docs
AbstractNetwork
HyperParams
hyperparams
forward
train!
set_test_mode!
params
regularized_params
```

### Conversion and copy

```@docs
Base.copy(::AbstractNetwork)
to_gpu
to_cpu
on_gpu
convert_input
convert_output
```

### Misc

```@docs
gc
```

## Derived functions

### Evaluation function

```@docs
evaluate
```

### Oracle interface

```@docs
MCTS.evaluate(::AbstractNetwork, board, actions)
MCTS.evaluate_batch(::AbstractNetwork, batch)
```

### Misc

```@docs
num_parameters
num_regularized_parameters
mean_weight
copy(::AbstractNetwork)
```
