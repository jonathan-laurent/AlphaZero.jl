#####
##### Neural Network Interface
#####

"""
  Network{Game}

Abstract type for a neural network oracle for `Game`.
It must implement the following interface
  - Flux.gpu(nn), Flux.cpu(nn)
  - Flux.params(nn), Flux.loadparams!(nn, p)
  - nn(boards, action_masks)
  - regularized_weights(nn)
  - num_parameters(nn)
  - network_report(nn)
"""
abstract type Network{G} <: MCTS.Oracle{G} end

function Base.copy(nn::Net) where Net <: Network
  new = Net()
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

@kwdef struct SimpleNetParams
  width :: Int = 300
  depth_common :: Int = 3
  depth_pbranch :: Int = 1
  depth_vbranch :: Int = 1
end

struct SimpleNet{G, params} <: Network{G}
  common
  vbranch
  pbranch
end

function SimpleNet{G, params}() where {G, params}
  @assert isa(params, SimpleNetParams)
  indim = GI.board_dim(G)
  outdim = GI.num_actions(G)
  hsize = params.width
  hlayers(depth) = [Dense(hsize, hsize, relu) for i in 1:depth]
  common = Chain(
    Dense(indim, hsize, relu),
    hlayers(params.depth_common)...)
  vbranch = Chain(
    hlayers(params.depth_vbranch)...,
    Dense(hsize, 1, tanh))
  pbranch = Chain(
    hlayers(params.depth_pbranch)...,
    Dense(hsize, outdim),
    softmax)
  SimpleNet{G, params}(common, vbranch, pbranch)
end

# Flux.@treelike does not work do to Network being parametric
Flux.children(nn::SimpleNet) = (nn.common, nn.vbranch, nn.pbranch)

function Flux.mapchildren(f, nn::Net) where Net <: SimpleNet
  Net(f(nn.common), f(nn.vbranch), f(nn.pbranch))
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
  f32(arr) = convert(AbstractArray{Float32}, arr)
  return f32.((W, X, A, P, V))
end

#####
##### Learning procedure
#####

# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
# Surprisingly, Flux does not like the code above (scalar operations)
mse_wmean(ŷ, y, w) = sum((ŷ .- y) .* (ŷ .- y) .* w) / sum(w)

klloss_wmean(π̂, π, w) = -sum(π .* log.(π̂ .+ eps(eltype(π))) .* w) / sum(w)

entropy_wmean(π, w) = -sum(π .* log.(π .+ eps(eltype(π))) .* w) / sum(w)

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
# Works with Float32
# Takes care of interfacing with the GPU
struct Trainer
  network
  examples
  samples
  Wmean
  Hp
  optimizer
  batch_size
  function Trainer(
    network::Network{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams
  ) where G
    examples = merge_by_board(examples)
    samples = convert_samples(G, examples)
    network = copy(network)
    if params.use_gpu
      CuArrays.allowscalar(false) # Does not work if moved to AlphaZero.jl
      samples = gpu.(samples)
      network = gpu(network)
    end
    W, X, A, P, V = samples
    Wmean = mean(W)
    Hp = entropy_wmean(P, W)
    optimizer = Flux.ADAM(params.learning_rate)
    batch_size = params.batch_size
    return new(network, examples, samples, Wmean, Hp, optimizer, batch_size)
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

function loss_report(tr::Trainer)
  ltuple = Tracker.data.(losses(tr.network, tr.Wmean, tr.Hp, tr.samples))
  return Report.Loss(ltuple...)
end

function learning_status(tr::Trainer)
  loss = loss_report(tr)
  net = network_report(tr.network)
  return Report.LearningStatus(loss, net)
end

function network_output_entropy(tr::Trainer)
  W, X, A, P, V = tr.samples
  P̂, _ = tr.network(X, A)
  return entropy_wmean(P̂, W) |> Tracker.data
end

function samples_report(tr::Trainer)
  loss = loss_report(tr)
  Hp = tr.Hp
  Hp̂ = network_output_entropy(tr)
  num_examples = sum(e.n for e in tr.examples)
  num_boards = length(tr.examples)
  Wtot = num_boards * tr.Wmean
  return Report.Samples(num_examples, num_boards, Wtot, loss, Hp, Hp̂)
end

function memory_report(
    mem::MemoryBuffer,
    nn::Network{G},
    params::LearningParams,
    nstages
    ) where G
  Tr(es) = Trainer(nn, es, params)
  latest_batch = samples_report(Tr(last_batch(mem)))
  all_samples = samples_report(Tr(get(mem)))
  per_game_stage = begin
    es = get(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / nstages)
    stages = collect(Iterators.partition(es, csize))
    map(stages) do es
      t = mean(e.t for e in es)
      stats = samples_report(Tr(es))
      (t, stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
