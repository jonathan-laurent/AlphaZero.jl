#####
##### Neural Network Interface
#####

"""
  Network{Game}

Abstract type for a neural network oracle for `Game`.
It must implement the following interface
  - Flux.gpu(nn), Flux.cpu(nn)
  - Flux.params(nn), Flux.loadparams!(nn, p)
  - hyperparams(nn)
  - HyperParams(typeof(nn))
  - nn(boards, action_masks)
  - regularized_weights(nn)
  - num_parameters(nn)
  - network_report(nn)
"""
abstract type Network{G} <: MCTS.Oracle{G} end

function Base.copy(nn::Net) where Net <: Network
  new = Net(hyperparams(nn))
  Flux.loadparams!(new, Flux.params(nn))
  return new
end

function MCTS.evaluate(nn::Network{G}, board, available_actions) where G
  mask = GI.actions_mask(G, available_actions)
  input = GI.vectorize_board(G, board)
  P, V = nn(input, mask)
  P = P[mask]
  return Tracker.data(P), Tracker.data(V)[1]
end

function MCTS.evaluate_batch(nn::Network{G}, batch) where G
  X = Util.concat_columns((GI.vectorize_board(G, b) for (b, as) in batch))
  A = Util.concat_columns((GI.actions_mask(G, as) for (b, as) in batch))
  P, V = Tracker.data.(nn(X, A))
  return [(P[A[:,i],i], V[1,i]) for i in eachindex(batch)]
end

#####
##### A simple example network
#####

@kwdef struct SimpleNetHyperParams
  width :: Int = 300
  depth_common :: Int = 3
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
end

struct SimpleNet{G} <: Network{G}
  hyper
  common
  vbranch
  pbranch
end

function SimpleNet{G}(hyper::SimpleNetHyperParams) where G
  indim = GI.board_dim(G)
  outdim = GI.num_actions(G)
  hsize = hyper.width
  hlayers(depth) = [Dense(hsize, hsize, relu) for i in 1:depth]
  common = Chain(
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

HyperParams(::Type{<:SimpleNet}) = SimpleNetHyperParams

hyperparams(nn::SimpleNet) = nn.hyper

# Flux.@treelike does not work do to Network being parametric
Flux.children(nn::SimpleNet) = (nn.common, nn.vbranch, nn.pbranch)

function Flux.mapchildren(f, nn::Net) where Net <: SimpleNet
  Net(nn.hyper, f(nn.common), f(nn.vbranch), f(nn.pbranch))
end

# Forward pass
function (nn::SimpleNet)(board, actions_mask)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c) .* actions_mask
  sp = sum(p, dims=1)
  @assert all(sp .> 0)
  p = p ./ sp
  return (p, v)
end

function regularized_weights(nn::SimpleNet)
  W(mlp) = [l.W for l in mlp if isa(l, Dense)]
  return [W(nn.common); W(nn.vbranch); W(nn.pbranch)]
end

function network_report(nn::SimpleNet) :: Report.Network
  Ws = Tracker.data.(regularized_weights(nn))
  maxw = maximum(maximum(abs.(W)) for W in Ws)
  meanw = mean(mean(abs.(W)) for W in Ws)
  pbiases = nn.pbranch[end-1].b |> Tracker.data
  vbias = nn.vbranch[end].b |> Tracker.data
  return Report.Network(maxw, meanw, pbiases, sum(vbias))
end

num_parameters(nn::SimpleNet) = sum(length(p) for p in Flux.params(nn))
