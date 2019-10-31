#####
##### Simple Example Network: Vanilla MLP
#####

@kwdef struct SimpleNetHP
  width :: Int
  depth_common :: Int
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
  use_batch_norm :: Bool = false
end

Util.generate_update_constructor(SimpleNetHP) |> eval

struct SimpleNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function SimpleNet{G}(hyper::SimpleNetHP) where G
  make_dense(indim, outdim) =
    hyper.use_batch_norm ?
      Chain(Dense(indim, outdim), Flux.BatchNorm(outdim, relu, momentum=1f0)) :
      Dense(indim, outdim, relu)
  indim = prod(GameInterface.board_dim(G))
  outdim = GameInterface.num_actions(G)
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  common = Chain(
    linearize,
    make_dense(indim, hsize),
    hlayers(hyper.depth_common)...)
  vbranch = Chain(
    hlayers(hyper.depth_vbranch)...,
    Dense(hsize, 1, tanh))
  pbranch = Chain(
    hlayers(hyper.depth_pbranch)...,
    Dense(hsize, outdim),
    softmax)
  SimpleNet{G}(hyper, common, vbranch, pbranch)
end

Network.HyperParams(::Type{<:SimpleNet}) = SimpleNetHP
