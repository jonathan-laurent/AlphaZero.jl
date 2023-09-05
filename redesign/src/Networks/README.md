# Network Library Structure

- [Network.jl](Network.jl): Contains the `Network` module, which defines the `FluxNetwork` interface.
    This interface defines the way to create Neural Networks with [Flux](https://fluxml.ai/Flux.jl/stable/)
    that can be used during the self-play and training phases of AlphaZero. Any derived class of networks
    must follow this interface, and can be included at the end of the [Network.jl](Network.jl) file
    and exported at the top, like currently done with the MLP and Res MLP architectures.
- [SimpleNet.jl](SimpleNet.jl): Implements the `SimpleNet` structure, which is a simple Multi-Layer
    Perceptron (MLP) architecture that follows the `FluxNetwork` interface.
- [SimpleResNet.jl](SimpleResNet.jl): Implements the `SimpleResNet` structure, which is a simple
    **Residual** MLP architecture that follows the `FluxNetwork` interface.