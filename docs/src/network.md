# Network Interface

```@meta
CurrentModule = AlphaZero.Network
```

```@docs
AbstractNetwork
HyperParams
hyperparams
forward
train!
regularized_weights
```

## Conversion and Copy

```@docs
Base.copy(::Network)
to_gpu
to_cpu
on_gpu
convert_input
convert_output
```

## Debugging and Profiling

```@docs
num_parameters
network_report
```
