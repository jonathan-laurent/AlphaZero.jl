# [Networks Library](@id networks_library)

```@meta
CurrentModule = AlphaZero
```

For convenience, we provide a library of standard networks implementing the
[neural network interface](@ref network_interface).

These networks are contained in the `AlphaZero.NetLib` module, which is resolved to
either `AlphaZero.KnetLib` or `AlphaZero.FluxLib` during precompilation depending on
the value of the `ALPHAZERO_DEFAULT_DL_FRAMEWORK` environment variable
(Knet is recommended and used by default).

## [Convolutional ResNet](@id conv_resnet)

```@docs
NetLib.ResNet
NetLib.ResNetHP
```

## [Simple Network](@id simplenet)

```@docs
NetLib.SimpleNet
NetLib.SimpleNetHP
```