#####
##### Neural Network Interface
#####

"""
  Network{Game}

Abstract type for a neural network oracle for `Game`.
It must implement the following interface
  - Flux.gpu(nn), Flux.cpu(nn)
  - Flux.params(nn), Flux.loadparams!(nn, p)
  - Network{Game}()
  - nn(boards, action_masks)
  - regularized_weights(nn)
  - num_parameters(nn)
  - network_report(nn)
"""
abstract type Network{G} <: MCTS.Oracle{G} end

function Base.copy(nn::Network{G}) where G
  new = Network{G}()
  Flux.loadparams!(new, Flux.params(nn))
  return new
end

function MCTS.evaluate(nn::Network{G}, board, available_actions) where G
  mask = GI.actions_mask(G, available_actions)
  input = GI.vectorize_board(G, board)
  P, V = nn(input, mask)
  P = P[mask]
  # The Float64 conversion is important in ase inference is done on a GPU
  # (therefore using Float32 numbers)
  return Vector{Float64}(Tracker.data(P)), Float64(Tracker.data(V)[1])
end

#####
##### A simple example network
#####

const SIMPLE_NET_HSIZE = 300

struct SimpleNet{G} <: Network{G}
  common
  vbranch
  pbranch
  function SimpleNet{G}() where G
    indim = GI.board_dim(G)
    outdim = GI.num_actions(G)
    hsize = SIMPLE_NET_HSIZE
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

Flux.@treelike SimpleNet

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
  return Report.Network(maxw, meanw, pbiases, vbias[1])
end

num_parameters(nn::SimpleNet) = sum(length(p) for p in Flux.params(nn))

#####
##### Converting samples
#####

function convert_sample(Game, e::TrainingExample)
  w = [log2(e.n) + 1]
  x = GI.vectorize_board(Game, e.b)
  actions = GI.available_actions(Game(e.b))
  a = GI.actions_mask(Game, actions)
  p = zeros(size(a))
  p[[GI.action_id(Game, a) for a in actions]] = e.π
  v = [e.z]
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

function losses(nn, Wmean, Hp, (W, X, A, P, V))
  P̂, V̂ = nn(X, A)
  C = mean(W) / Wmean
  Lp = C * (klloss_wmean(P̂, P, W) - Hp)
  Lv = C * mse_wmean(V̂, V, W)
  Lreg = zero(Lv)
  L = Lp + Lv + Lreg
  return (L, Lp, Lv, Lreg)
end

# Does not mody the network it is given in place.
struct Trainer
  network
  samples
  Wmean
  Hp
  optimizer
  batch_size
  function Trainer(
    network::Network{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams;
    use_gpu=false
  ) where G
    samples = convert_samples(G, examples)
    network = copy(network)
    if use_gpu
      samples = gpu.(samples)
      network = gpu.(network)
    end
    W, X, A, P, V = samples
    Wmean = mean(W)
    Hp = entropy_wmean(P, W)
    optimizer = Flux.ADAM(params.learning_rate)
    batch_size = params.batch_size
    return new(network, samples, Wmean, Hp, optimizer, batch_size)
  end
end

get_trained_network(tr::Trainer) = tr.network |> copy |> cpu

function training_epoch!(tr::Trainer)
  loss(batch...) = losses(tr.network, tr.Wmean, tr.Hp, batch)[1]
  data = Util.random_batches(tr.samples, tr.batch_size)
  Flux.train!(loss, Flux.params(tr.network), data, tr.optimizer)
end

#####
##### Generating debugging reports
#####

function loss_report(network, samples)
  W, X, A, P, V = samples
  Wmean = mean(W)
  Hp = entropy_wmean(P, W)
  ltuple = Tracker.data.(losses(network, Wmean, Hp, samples))
  return Report.Loss(ltuple...)
end

loss_report(tr::Trainer) = loss_report(tr.network, tr.samples)

function learning_status(tr::Trainer)
  loss = loss_report(tr)
  net = network_report(tr.network)
  return Report.LearningStatus(loss, net)
end

# Note: this analysis has to be done on GPU.

function analyze_samples(
    examples::Vector{<:TrainingExample}, # assume not merged yet
    network::Network{G}
  ) :: Report.SamplesSub where G
  examples = merge_per_board(examples)
  samples = convert_samples(G, examples)
  loss = loss_report(network, samples)
  W, X, A, P, V = samples
  P̂, V̂ = Tracker.data.(network.nn(X, A))
  Hp = entropy_wmean(P, W)
  Hp̂ = entropy_wmean(P̂, W)
  Wtot = sum(W)
  num_samples = sum(e.n for e in examples)
  num_boards = length(examples)
  return Report.SamplesSub(num_samples, num_boards, Wtot, loss, Hp, Hp̂)
end

function analyze_samples(mem::MemoryBuffer, nn::Network{G}, nstages) where G
  latest_batch = analyze_samples(last_batch_raw(mem), nn)
  all_samples = analyze_samples(get_raw(mem), nn)
  per_game_stage = begin
    es = get_raw(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / nstages)
    stages = collect(Iterators.partition(es, csize))
    map(stages) do es
      t = mean(e.t for e in es)
      stats = analyze_samples(es, nn)
      (t, stats)
    end
  end
  return Report.Samples(latest_batch, all_samples, per_game_stage)
end
