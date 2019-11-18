using AlphaZero
using Revise

include("../games/connect-four/game.jl")
using .ConnectFour

network = ResNet{Game}(ResNetHP(
  num_filters=128,
  num_blocks=10,
  conv_kernel_size=(3,1),
  num_policy_head_filters=4,
  num_value_head_filters=32,
  batch_norm_momentum=0.3))

network = Network.copy(network, on_gpu=true, test_mode=false)

@show AlphaZero.Network.num_parameters(network)

lp = LearningParams(
  batch_size=256,
  loss_computation_batch_size=1024,
  gc_every=2_000,
  learning_rate=1e-3,
  l2_regularization=1e-4,
  nonvalidity_penalty=1.,
  checkpoints=[1, 2, 4])

f32bool = [0f0, 1f0]
na = GI.num_actions(Game)
bdim = GI.board_dim(Game)

function genbatch(size)
  W = ones(Float32, 1, size)
  X = rand(f32bool,bdim..., size)
  A = rand(f32bool, na, size)
  P = ones(Float32, na, size) ./ na
  V = zeros(Float32, 1, size)
  return (W, X, A, P, V)
end

using ProgressMeter

function looploss(batch_size, n_batches)
  batch = genbatch(batch_size)
  @progress for i in 1:n_batches
    Ls = AlphaZero.losses(
      network,
      lp, 1f0, 1f0,
      Network.convert_input_tuple(network, batch))
  end
end

for i in 1:10
  looploss(1024, 100)
  looploss(64, 1000)
end
