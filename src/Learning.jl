#####
##### Defining the canonical network architecture
#####

# Numeric type used for learning (Float32 faster on GPUs)
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

function regularized_weights(nn::Network)
  W(mlp) = [l.W for l in mlp if isa(l, Dense)]
  return [W(nn.common); W(nn.vbranch); W(nn.pbranch)]
end

function network_report(nn::Network) :: Report.Network
  Ws = Tracker.data.(regularized_weights(nn))
  maxw = maximum(maximum(abs.(W)) for W in Ws)
  meanw = mean(mean(abs.(W)) for W in Ws)
  pbiases = nn.pbranch[end-1].b |> Tracker.data
  vbias = nn.vbranch[end].b |> Tracker.data
  return Report.Network(maxw, meanw, pbiases, vbias[1])
end

num_parameters(nn::Network) = sum(length(p) for p in Flux.params(nn))

#####
##### Oracle: bridge between MCTS and the neural network
#####

struct Oracle{Game} <: MCTS.Oracle{Game}
  nn :: Network
  function Oracle{G}() where G
    hsize = 300
    nn = Network(GI.board_dim(G), GI.num_actions(G), hsize)
    new{G}(nn)
  end
end

function Base.copy(o::Oracle{G}) where G
  new = Oracle{G}()
  Flux.loadparams!(new.nn, Flux.params(o.nn))
  return new
end

num_parameters(o::Oracle) = num_parameters(o.nn)

function MCTS.evaluate(o::Oracle{G}, board, available_actions) where G
  mask = GI.actions_mask(G, available_actions)
  input = GI.vectorize_board(G, R, board)
  P, V = o.nn(input, mask)
  P = P[mask]
  return Vector{Float64}(Tracker.data(P)), Float64(Tracker.data(V)[1])
end

#####
##### Converting samples
#####

function convert_sample(Game, e::TrainingExample)
  w = [log2(R(e.n)) + one(R)]
  x = Vector{R}(GI.vectorize_board(Game, R, e.b))
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

#####
##### Learning procedure
#####

mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) ./ sum(w)

klloss_wmean(π̂, π, w) = -sum(π .* log.(π̂ .+ eps(eltype(π))) .* w) ./ sum(w)

entropy_wmean(π, w) = -sum(π .* log.(π .+ eps(eltype(π))) .* w) ./ sum(w)

function losses(oracle, Wmean, (W, X, A, P, V))
  P̂, V̂ = oracle.nn(X, A)
  C = mean(W) / Wmean
  Lp = C * klloss_wmean(P̂, P, W)
  Lv = C * mse_wmean(V̂, V, W)
  Lreg = zero(Lv)
  return (Lp, Lv, Lreg)
end

struct Trainer
  oracle
  samples
  Wmean
  optimizer
  batch_size
  function Trainer(
    oracle::Oracle{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams
  ) where G
    samples = convert_samples(G, examples)
    W, X, A, P, V = samples
    Wmean = mean(W)
    optimizer = Flux.ADAM(params.learning_rate)
    batch_size = params.batch_size
    return new(oracle, samples, Wmean, optimizer, batch_size)
  end
end

function training_epoch!(tr::Trainer)
  loss(batch...) = sum(losses(tr.oracle, tr.Wmean, batch))
  data = Util.random_batches(tr.samples, tr.batch_size)
  Flux.train!(loss, Flux.params(tr.oracle.nn), data, tr.optimizer)
end

function loss_report(tr::Trainer)
  Lp, Lv, Lreg = Tracker.data.(losses(tr.oracle, tr.Wmean, tr.samples))
  L = Lp + Lv + Lreg
  return Report.Loss(L, Lp, Lv, Lreg)
end

function learning_status(tr::Trainer)
  loss = loss_report(tr)
  net = network_report(tr.oracle.nn)
  return Report.LearningStatus(loss, net)
end
