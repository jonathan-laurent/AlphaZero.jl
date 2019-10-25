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

wmean(x, w) = sum(x .* w) / sum(w)

function losses(nn, params, Wmean, Hp, (W, X, A, P, V))
  creg = params.l2_regularization
  cinv = params.nonvalidity_penalty
  P̂, V̂, p_invalid = Networks.evaluate(nn, X, A)
  Lp = klloss_wmean(P̂, P, W) - Hp
  Lv = mse_wmean(V̂, V, W)
  Lreg = iszero(creg) ?
    zero(Lv) :
    creg * sum(sum(w .* w) for w in regularized_weights(nn))
  Linv = iszero(cinv) ?
    zero(Lv) :
    cinv * wmean(p_invalid, W)
  L = (mean(W) / Wmean) * (Lp + Lv + Lreg + Linv)
  return (L, Lp, Lv, Lreg, Linv)
end

# Does not mody the network it is given in place.
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
    network::Network{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams
  ) where G
    examples = merge_by_board(examples)
    samples = convert_samples(G, examples)
    network = copy(network)
    network = params.use_gpu ?
      Networks.to_gpu(network) : Networks.to_cpu(network)
    samples = Networks.convert_input_tuple(network, samples)
    W, X, A, P, V = samples
    Wmean = mean(W)
    Hp = entropy_wmean(P, W)
    return new(network, examples, params, samples, Wmean, Hp)
  end
end

get_trained_network(tr::Trainer) = tr.network |> copy |> Networks.to_cpu

function training_epoch!(tr::Trainer)
  loss(batch...) = losses(
    tr.network, tr.params, tr.Wmean, tr.Hp, batch)[1]
  data = Util.random_batches(tr.samples, tr.params.batch_size)
  Networks.train!(tr.network, loss, data, tr.params.learning_rate)
end

#####
##### Generating debugging reports
#####

function loss_report(tr::Trainer)
  ltuple = Networks.convert_output_tuple(tr.network,
    losses(tr.network, tr.params, tr.Wmean, tr.Hp, tr.samples))
  return Report.Loss(ltuple...)
end

function learning_status(tr::Trainer)
  loss = loss_report(tr)
  net = Networks.network_report(tr.network)
  return Report.LearningStatus(loss, net)
end

function network_output_entropy(tr::Trainer)
  W, X, A, P, V = tr.samples
  P̂, _ = Networks.evaluate(tr.network, X, A)
  return Networks.convert_output(tr.network, entropy_wmean(P̂, W))
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
      Report.StageSamples(t, stats)
    end
  end
  return Report.Memory(latest_batch, all_samples, per_game_stage)
end
