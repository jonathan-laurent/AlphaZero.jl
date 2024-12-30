"""
    SimpleResNetHP

Hyperparameters for the simple Resnet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width::Int`                  | Number of neurons on each dense layer        |
| `depth_common::Int`           | Number of dense layers in the trunk          |
| `depth_phead::Int = 1`        | Number of hidden layers in the actions head  |
| `depth_vhead::Int = 1`        | Number of hidden layers in the value  head   |
"""
@kwdef struct SimpleResNetHP
    width::Int
    depth_common::Int
    depth_phead::Int = 1
    depth_vhead::Int = 1
end


"""
    SimpleResNet <: FluxNetwork

A simple two-headed architecture with only dense layers and skip connections.
"""
mutable struct SimpleResNet <: FluxNetwork
    hyper
    common
    vhead
    phead
end
Flux.@functor SimpleResNet (common, vhead, phead)


function SimpleResNet(indim::Int, outdim::Int, hyper::SimpleResNetHP)
    rng = MersenneTwister(3409)
    weight_init() = Flux.glorot_uniform(rng)

    make_dense(indim, outdim) = Flux.Dense(indim => outdim, Flux.relu; init=weight_init())
    function make_skip_dense(indim, outdim)
        return Flux.Chain(Flux.SkipConnection(make_dense(indim, outdim), +), Flux.relu)
    end

    hidden_skip_layers(depth) = [make_skip_dense(hyper.width, hyper.width) for _ in 1:depth]

    common = Flux.Chain(
        Flux.flatten,
        make_dense(indim, hyper.width),
        hidden_skip_layers(hyper.depth_common)...
    )

    vhead = Flux.Chain(
        hidden_skip_layers(hyper.depth_vhead)...,
        Flux.Dense(hyper.width => 1, tanh; init=weight_init())
    )

    phead = Flux.Chain(
        hidden_skip_layers(hyper.depth_phead)...,
        Flux.Dense(hyper.width => outdim; init=weight_init())
    )

    SimpleResNet(hyper, common, vhead, phead)
end


HyperParams(::Type{SimpleResNet}) = SimpleResNetHP
hyperparams(nn::SimpleResNet) = nn.hyper
on_gpu(nn::SimpleResNet) = arr_is_on_gpu(nn.vhead[end].bias)


function forward(nn::SimpleResNet, states, use_softmax=false)
    common = nn.common(states)
    v = nn.vhead(common)
    p = nn.phead(common)
    use_softmax && (p = Flux.softmax(p))
    return v, p
end
