#####
##### Dense Resnet
#####

# Recommended configuration:
# AlphaZero: 20 blocks (40 in final version), 256 filters
# Oracle basic: 5 blocks, 64 filters
# Oracle final: 20 blocks, 128 filters
@kwdef struct ResNetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_policy_head_filters :: Int = 2
  num_value_head_filters :: Int = 1
  batch_norm_momentum :: Float32 = 1f0
end

Util.generate_update_constructor(ResNetHP) |> eval

mutable struct ResNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
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

function ResNet{G}(hyper::ResNetHP) where G
  bsize = GameInterface.board_dim(G)
  outdim = GameInterface.num_actions(G)
  ksize = hyper.conv_kernel_size
  @assert all(ksize .% 2 .== 1)
  pad = ksize .÷ 2
  nf = hyper.num_filters
  npf = hyper.num_policy_head_filters
  nvf = hyper.num_value_head_filters
  bnmom = hyper.batch_norm_momentum
  common = Chain(
    Conv(ksize, bsize[3]=>nf, pad=pad),
    BatchNorm(nf, relu, momentum=bnmom),
    [ResNetBlock(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
  pbranch = Chain(
    Conv((1, 1), nf=>npf),
    BatchNorm(npf, relu, momentum=bnmom),
    linearize,
    Dense(bsize[1] * bsize[2] * npf, outdim),
    softmax)
  vbranch = Chain(
    Conv((1, 1), nf=>nvf),
    BatchNorm(nvf, relu, momentum=bnmom),
    linearize,
    Dense(bsize[1] * bsize[2] * nvf, nf, relu),
    Dense(nf, 1, tanh))
  ResNet{G}(hyper, common, vbranch, pbranch)
end

Network.HyperParams(::Type{<:ResNet}) = ResNetHP
