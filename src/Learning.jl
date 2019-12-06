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

# mse_wmean(ŷ, y, w) = sum((ŷ .- y).^2 .* w) / sum(w)
# Surprisingly, Flux does not like the code above (scalar operations)
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

# Works with Float32
# Takes care of interfacing with the GPU
struct Trainer
  network
  examples
  params
  samples
  Wmean
  Hp
  function Trainer(
    network::AbstractNetwork{G},
    examples::AbstractVector{<:TrainingExample},
    params::LearningParams
  ) where G
    examples = merge_by_board(examples)
    samples = convert_samples(G, examples)
    network = Network.copy(network, on_gpu=params.use_gpu, test_mode=false)
    #samples = Network.convert_input_tuple(network, samples)
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
  data = Util.periodic_gc(data, tr.params.gc_every) do
    Network.gc(tr.network)
  end
  Network.train!(tr.network, loss, data, tr.params.learning_rate)
  Network.gc(tr.network)
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
  batches = Util.periodic_gc(batches, tr.params.gc_every) do
    Network.gc(tr.network)
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
  return Report.Samples(num_examples, num_boards, Wtot, tr.Hp, status)
end

function memory_report(
    mem::MemoryBuffer,
    nn::AbstractNetwork{G},
    params::LearningParams,
    nstages
    ) where G
  Tr(es) = Trainer(nn, es, params)
  all_samples = samples_report(Tr(get(mem)))
  latest_batch = isempty(last_batch(mem)) ?
    all_samples :
    samples_report(Tr(last_batch(mem)))
  per_game_stage = begin
    es = get(mem)
    sort!(es, by=(e->e.t))
    csize = ceil(Int, length(es) / nstages)
    stages = collect(Iterators.partition(es, csize))
    map(stages) do es
      t = mean(e.t for e in es)
      stats = samples_report(Tr(es))
      Report.StageSamples(t, stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
