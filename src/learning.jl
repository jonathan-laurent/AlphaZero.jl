#####
##### Converting samples
#####

# A samples collection is represented on the learning side as a (W, X, A, P, V)
# tuple. Each component is a `Float32` tensor whose last dimension corresponds
# to the sample index. Writing `n` the number of samples and `a` the total
# number of actions:
# - W (size 1×n) contains the samples weights
# - X (size …×n) contains the board representations
# - A (size a×n) contains the action masks (values are either 0 or 1)
# - P (size a×n) contains the recorded MCTS policies
# - V (size 1×n) contains the recorded values
# Note that the weight of a sample is computed as an increasing
# function of its `n` field.

function convert_sample(Game, wp, e::TrainingSample)
  if wp == CONSTANT_WEIGHT
    w = Float32[1]
  elseif wp == LOG_WEIGHT
    w = Float32[log2(e.n) + 1]
  else
    @assert wp == LINEAR_WEIGHT
    w = Float32[n]
  end
  x = GI.vectorize_board(Game, e.b)
  a = GI.actions_mask(Game(e.b))
  p = zeros(size(a))
  p[a] = e.π
  v = [e.z]
  return (w, x, a, p, v)
end

function convert_samples(Game, wp, es::Vector{<:TrainingSample})
  ces = [convert_sample(Game, wp, e) for e in es]
  W = Util.superpose((e[1] for e in ces))
  X = Util.superpose((e[2] for e in ces))
  A = Util.superpose((e[3] for e in ces))
  P = Util.superpose((e[4] for e in ces))
  V = Util.superpose((e[5] for e in ces))
  f32(arr) = convert(AbstractArray{Float32}, arr)
  return f32.((W, X, A, P, V))
end

#####
##### Learning procedure
#####

# Surprisingly, Flux does not like the following code (scalar operations):
# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
mse_wmean(ŷ, y, w) = sum((ŷ .- y) .* (ŷ .- y) .* w) / sum(w)

klloss_wmean(π̂, π, w) = -sum(π .* log.(π̂ .+ eps(eltype(π))) .* w) / sum(w)

entropy_wmean(π, w) = -sum(π .* log.(π .+ eps(eltype(π))) .* w) / sum(w)

wmean(x, w) = sum(x .* w) / sum(w)

function losses(nn, params, Wmean, Hp, (W, X, A, P, V))
  creg = params.l2_regularization
  cinv = params.nonvalidity_penalty
  P̂, V̂, p_invalid = Network.evaluate(nn, X, A)
  Lp = klloss_wmean(P̂, P, W) - Hp
  Lv = mse_wmean(V̂, V, W)
  Lreg = iszero(creg) ?
    zero(Lv) :
    creg * sum(sum(w .* w) for w in Network.regularized_params(nn))
  Linv = iszero(cinv) ?
    zero(Lv) :
    cinv * wmean(p_invalid, W)
  L = (mean(W) / Wmean) * (Lp + Lv + Lreg + Linv)
  return (L, Lp, Lv, Lreg, Linv)
end

struct Trainer
  network
  examples
  params
  samples
  Wmean
  Hp
  function Trainer(
    network::AbstractNetwork{G},
    examples::AbstractVector{<:TrainingSample},
    params::LearningParams
  ) where G
    examples = merge_by_board(examples)
    samples = convert_samples(G, params.samples_weighing_policy, examples)
    network = Network.copy(network, on_gpu=params.use_gpu, test_mode=false)
    W, X, A, P, V = samples
    Wmean = mean(W)
    Hp = entropy_wmean(P, W)
    return new(network, examples, params, samples, Wmean, Hp)
  end
end

function get_trained_network(tr::Trainer)
  Network.copy(tr.network, on_gpu=false, test_mode=true)
end

function training_epoch!(tr::Trainer)
  loss(batch...) = losses(
    tr.network, tr.params, tr.Wmean, tr.Hp, batch)[1]
  data = Util.random_batches(tr.samples, tr.params.batch_size) do x
    Network.convert_input(tr.network, x)
  end
  if !isnothing(tr.params.gc_every)
    data = Util.periodic_gc(data, tr.params.gc_every) do
      Network.gc(tr.network)
    end
  end
  ls = Vector{Float32}()
  Network.train!(tr.network, tr.params.optimiser, loss, data) do i, l
    push!(ls, l)
  end
  Network.gc(tr.network)
  return ls
end

#####
##### Generating debugging reports
#####

function mean_learning_status(reports::Vector{Report.LearningStatus})
  L     = mean(r.loss.L     for r in reports)
  Lp    = mean(r.loss.Lp    for r in reports)
  Lv    = mean(r.loss.Lv    for r in reports)
  Lreg  = mean(r.loss.Lreg  for r in reports)
  Linv  = mean(r.loss.Linv  for r in reports)
  Hpnet = mean(r.Hpnet      for r in reports)
  Hp    = mean(r.Hp         for r in reports)
  return Report.LearningStatus(Report.Loss(L, Lp, Lv, Lreg, Linv), Hp, Hpnet)
end

function learning_status(tr::Trainer, samples)
  # As done now, this is slighly inefficient as we solve the
  # same neural network inference problem twice
  W, X, A, P, V = samples
  Ls = losses(tr.network, tr.params, tr.Wmean, tr.Hp, samples)
  Ls = Network.convert_output_tuple(tr.network, Ls)
  Pnet, _ = Network.evaluate(tr.network, X, A)
  Hpnet = entropy_wmean(Pnet, W)
  Hpnet = Network.convert_output(tr.network, Hpnet)
  return Report.LearningStatus(Report.Loss(Ls...), tr.Hp, Hpnet)
end

function learning_status(tr::Trainer)
  batch_size = tr.params.loss_computation_batch_size
  # If there are less samples
  partial = size(tr.samples[1])[end] < batch_size
  batches = Util.random_batches(tr.samples, batch_size, partial=partial) do x
    Network.convert_input(tr.network, x)
  end
  if !isnothing(tr.params.gc_every)
    batches = Util.periodic_gc(batches, tr.params.gc_every) do
      Network.gc(tr.network)
    end
  end
  reports = [learning_status(tr, batch) for batch in batches]
  Network.gc(tr.network)
  return mean_learning_status(reports)
end

function samples_report(tr::Trainer)
  status = learning_status(tr)
  num_examples = sum(e.n for e in tr.examples)
  num_boards = length(tr.examples)
  Wtot = num_boards * tr.Wmean
  return Report.Samples(num_examples, num_boards, Wtot, status)
end

function memory_report(
    mem::MemoryBuffer,
    nn::AbstractNetwork{G},
    learning_params::LearningParams,
    params::MemAnalysisParams
    ) where G
  Tr(es) = Trainer(nn, es, learning_params)
  all_samples = samples_report(Tr(get(mem)))
  latest_batch = isempty(last_batch(mem)) ?
    all_samples :
    samples_report(Tr(last_batch(mem)))
  per_game_stage = begin
    es = get(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / params.num_game_stages)
    stages = collect(Iterators.partition(es, csize))
    map(stages) do es
      t = mean(e.t for e in es)
      stats = samples_report(Tr(es))
      Report.StageSamples(t, stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
