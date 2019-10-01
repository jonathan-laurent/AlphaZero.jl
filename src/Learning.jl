################################################################################
# Learning.jl
################################################################################

import Flux
using Flux: Tracker, Chain, Dense, relu, softmax

const R = Float32

struct Network
  common
  vbranch
  pbranch
  function Network(indim, outdim, hsize)
    common = Chain(
      Dense(indim, hsize, relu),
      Dense(hsize, hsize, relu),
      Dense(hsize, hsize, relu),
      Dense(hsize, hsize, relu))
    vbranch = Chain(
      Dense(hsize, hsize, relu),
      Dense(hsize, 1, tanh))
    pbranch = Chain(
      Dense(hsize, hsize, relu),
      Dense(hsize, outdim),
      softmax)
    new(common, vbranch, pbranch)
  end
end

Flux.@treelike Network

# Define the forward pass
function (nn::Network)(board, actions_mask)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c) .* actions_mask
  sp = sum(p)
  @assert sp > 0
  p = p ./ sp
  return (p, v)
end

################################################################################
# Interface to the neural network

struct Oracle{Game} <: MCTS.Oracle
  nn :: Network
  function Oracle{G}() where G
    hsize = 100
    nn = Network(GI.board_dim(G), GI.num_actions(G), hsize)
    new(nn)
  end
end

num_parameters(oracle) = sum(length(p) for p in params(o.nn))

function Base.copy(o::Oracle{G}) where G
  new = Oracle{G}()
  Flux.loadparams!(new.nn, Flux.params(o.nn))
  return new
end

function MCTS.evaluate(o::Oracle{G}, board, available_actions) where G
  mask = GI.actions_mask(G, available_actions)
  input = GI.vectorize_board(G, board)
  P, V = o.nn(input, mask)
  P = P[mask]
  return Vector{Float64}(Tracker.data(P)), Float64(Tracker.data(V)[1])
end

################################################################################

function random_minibatch(W, X, A, P, V; batchsize)
  n = size(X, 2)
  indices = rand(1:n, batchsize)
  return (W[:,indices], X[:,indices], A[:,indices], P[:,indices], V[:,indices])
end

function convert_sample(Game, e::TrainingExample)
  w = [log2(R(e.n)) + one(R)]
  x = Vector{R}(GI.vectorize_board(Game, e.b))
  actions = GI.available_actions(Game(e.b))
  a = GI.actions_mask(Game, actions)
  p = zeros(R, size(a))
  p[[GI.action_id(Game, a) for a in actions]] = e.π
  v = [R(e.z)]
  return (w, x, a, p, v)
end

function convert_samples(Game, es::Vector{<:TrainingExample})
  ces = [convert_sample(Game, e) for e in es]
  W = Util.concat_columns((e[1] for e in ces))
  X = Util.concat_columns((e[2] for e in ces))
  A = Util.concat_columns((e[3] for e in ces))
  P = Util.concat_columns((e[4] for e in ces))
  V = Util.concat_columns((e[5] for e in ces))
  return (W, X, A, P, V)
end

function train!(
    oracle::Oracle{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams
  ) where G

  opt = Flux.ADAM(params.learning_rate)
  let (W, X, A, P, V) = convert_samples(G, examples)
  let prevloss = Util.infinity(R)
    function loss(W, X, A, P₀, V₀)
      let (P, V) = oracle.nn(X, A)
        Lp = Flux.crossentropy(P .+ eps(R), P₀, weight = W)
        Lv = Util.weighted_mse(V, V₀, W)
        return Lp + Lv
      end
    end
    function print_legend()
      @printf("%12s", "Loss\n")
    end
    function cb()
      L = loss(W, X, A, P, V)
      prevloss - L < params.loss_eps && Flux.stop()
      prevloss = L
      @printf("%12.7f\n", L)
    end
    data = (
      random_minibatch(W, X, A, P, V; batchsize=params.batch_size)
      for i in 1:params.num_batches)
    print_legend()
    Flux.train!(
      loss, Flux.params(oracle.nn), data, opt, cb=Flux.throttle(cb, 1.0))
  end end
end

################################################################################
