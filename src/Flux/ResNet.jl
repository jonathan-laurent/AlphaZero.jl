#####
##### Dense Resnet
#####

@kwdef struct ResNetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int} = (3, 3)
  num_policy_head_filters :: Int = 2
  num_value_head_filters :: Int = 1
end

Util.generate_update_constructor(ResNetHP) |> eval

const BASIC_RESNET_HP = ResNetHP(
  num_blocks = 5,
  num_filters = 64)

const ALPHAZERO_RESNET_HP = ResNetHP(
  num_blocks = 20,
  num_filters = 256)

struct ResNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function ResNetBlock(size, n)
  pad = size .÷ 2
  layers = Flux.Chain(
    Flux.Conv(size, n=>n, pad=pad),
    Flux.BatchNorm(n, relu, momentum=1f0),
    Flux.Conv(size, n=>n, pad=pad),
    Flux.BatchNorm(n, momentum=1f0))
  return Flux.Chain(
    Flux.SkipConnection(layers, +),
    x -> Flux.relu.(x))
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
  common = Flux.Chain(
    Flux.Conv(ksize, bsize[3]=>nf, pad=pad),
    Flux.BatchNorm(nf, relu, momentum=1f0),
    [ResNetBlock(ksize, nf) for i in 1:hyper.num_blocks]...)
  pbranch = Flux.Chain(
    Flux.Conv((1, 1), nf=>npf),
    Flux.BatchNorm(npf, relu, momentum=1f0),
    linearize,
    Dense(bsize[1] * bsize[2] * npf, outdim),
    softmax)
  vbranch = Flux.Chain(
    Flux.Conv((1, 1), nf=>nvf),
    Flux.BatchNorm(nvf, relu, momentum=1f0),
    linearize,
    Dense(bsize[1] * bsize[2] * nvf, nf, relu),
    Dense(nf, 1, tanh))
  ResNet{G}(hyper, common, vbranch, pbranch)
end

Network.HyperParams(::Type{<:ResNet}) = ResNetHP
