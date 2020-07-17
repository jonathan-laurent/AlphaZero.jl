# [Network Interface](@id network_interface)

```@meta
CurrentModule = AlphaZero.Network
```

```@docs
Network
```

## Mandatory Interface

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

### Conversion and Copy

```@docs
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

## Derived Functions

### Evaluation Function

```@docs
evaluate
evaluate_batch
```

### Oracle Interface

All subtypes of `AbstractNetwork` implement the
[`MCTS.Oracle`](@ref) interface along with [`evaluate_batch`].

Since evaluating a neural network on single samples at a
time is slow, the latter should be used whenever possible.

### Misc

```@docs
num_parameters
num_regularized_parameters
mean_weight
copy(::AbstractNetwork)
```

### Optimiser Specification

```@docs
OptimiserSpec
CyclicNesterov
Adam
```
