#####
##### Simple Example Network: Vanilla MLP
#####

@kwdef struct SimpleNetHP
  width :: Int
  depth_common :: Int
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
end

struct SimpleNet{Game} <: TwoHeadNetwork{Game}
  hyper
  common
  vbranch
  pbranch
end

function linearize(x)
  s = size(x)
  return reshape(x, prod(s[1:end-1]), s[end])
end

function SimpleNet{G}(hyper::SimpleNetHP) where G
  indim = prod(GameInterface.board_dim(G))
  outdim = GameInterface.num_actions(G)
  hsize = hyper.width
  hlayers(depth) = [Dense(hsize, hsize, relu) for i in 1:depth]
  common = Chain(
    linearize,
    Dense(indim, hsize, relu),
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
