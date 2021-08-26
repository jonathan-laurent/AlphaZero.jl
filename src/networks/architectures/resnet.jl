"""
    ResNetHP

Hyperparameters for the convolutional resnet architecture.

| Parameter                 | Type                | Default   |
|:--------------------------|:--------------------|:----------|
| `num_blocks`              | `Int`               |  -        |
| `num_filters`             | `Int`               |  -        |
| `conv_kernel_size`        | `Tuple{Int, Int}`   |  -        |
| `num_policy_head_filters` | `Int`               | `2`       |
| `num_value_head_filters`  | `Int`               | `1`       |
| `batch_norm_momentum`     | `Float32`           | `0.6f0`   |

The trunk of the two-head network consists of `num_blocks` consecutive blocks.
Each block features two convolutional layers with `num_filters` filters and
with kernel size `conv_kernel_size`. Note that both kernel dimensions must be
odd.

During training, the network is evaluated in training mode on the whole
dataset to compute the loss before it is switched to test model, using
big batches. Therefore, it makes sense to use a high batch norm momentum
(put a lot of weight on the latest measurement).

# AlphaGo Zero Parameters

The network in the original paper from Deepmind features 20 blocks with 256
filters per convolutional layer.
"""
@kwdef struct ResNetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_policy_head_filters :: Int = 2
  num_value_head_filters :: Int = 1
  batch_norm_momentum :: Float32 = 0.6f0
end

"""
    ResNet <: TwoHeadNetwork

The convolutional residual network architecture that is used
in the original AlphaGo Zero paper.
"""
mutable struct ResNet <: TwoHeadNetwork
  gspec
  hyper
  common
  vhead
  phead
end

function ResNetBlock(size, n, bnmom)
  pad = size .÷ 2
  layers = Chain(
    Conv(size, n=>n, pad=pad),
    BatchNorm(n, relu, momentum=bnmom),
    Conv(size, n=>n, pad=pad),
    BatchNorm(n, momentum=bnmom))
  return Chain(
    SkipConnection(layers, +),
    x -> relu.(x))
end

function ResNet(gspec::AbstractGameSpec, hyper::ResNetHP)
  indim = GI.state_dim(gspec)
  outdim = GI.num_actions(gspec)
  ksize = hyper.conv_kernel_size
  @assert all(ksize .% 2 .== 1)
  pad = ksize .÷ 2
  nf = hyper.num_filters
  npf = hyper.num_policy_head_filters
  nvf = hyper.num_value_head_filters
  bnmom = hyper.batch_norm_momentum
  common = Chain(
    Conv(ksize, indim[3]=>nf, pad=pad),
    BatchNorm(nf, relu, momentum=bnmom),
    [ResNetBlock(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
  phead = Chain(
    Conv((1, 1), nf=>npf),
    BatchNorm(npf, relu, momentum=bnmom),
    flatten,
    Dense(indim[1] * indim[2] * npf, outdim),
    softmax)
  vhead = Chain(
    Conv((1, 1), nf=>nvf),
    BatchNorm(nvf, relu, momentum=bnmom),
    flatten,
    Dense(indim[1] * indim[2] * nvf, nf, relu),
    Dense(nf, 1, tanh))
  ResNet(gspec, hyper, common, vhead, phead)
end

Network.HyperParams(::Type{<:ResNet}) = ResNetHP

function Base.copy(nn::ResNet)
  return ResNet(
    nn.gspec,
    nn.hyper,
    deepcopy(nn.common),
    deepcopy(nn.vhead),
    deepcopy(nn.phead)
  )
end