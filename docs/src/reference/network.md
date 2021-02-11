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
game_spec
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

### Evaluation Functions

```@docs
forward_normalized
evaluate
evaluate_batch
```

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
