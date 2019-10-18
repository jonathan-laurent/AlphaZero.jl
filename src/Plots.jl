#####
##### Training summary plots
#####

function plot_report(
    params::Params,
    iterations::Vector{Report.Iteration},
    validation::Option{Vector{ValidationReport}},
    dir::String)
  n = length(iterations)
  isnothing(validation) || @assert length(validation) == n + 1
  iszero(n) && return
  isdir(dir) || mkpath(dir)
  plots, files = [], []
  # Validation score
  if !isnothing(validation)
    vbars = Plots.plot(0:n,
      [v.z for v in validation],
      title="Validation Score",
      #ylabel="Average reward",
      # ylims=(-1.0, 1.0),
      legend=nothing)
    push!(plots, vbars)
    push!(files, "validation")
  end
  prependz(x) = [0;x]
  duplast(x) = [x;x[end]]
  # Number of samples
  nsamples = Plots.plot(0:n,
    prependz([it.memory.all_samples.num_samples for it in iterations]),
    title="Experience Buffer Size",
    label="Number of samples")
  Plots.plot!(nsamples, 0:n,
    prependz([it.memory.all_samples.num_boards for it in iterations]),
    label="Number of distinct boards")
  # Performances during evaluation
  arena = Plots.plot(0:n,
    duplast([
      maximum(c.reward for c in it.learning.checkpoints)
      for it in iterations]),
    title="Arena Results",
    t=:bar,
    legend=:none)
  Plots.hline!(arena, [0, params.arena.update_threshold])
  # Loss on last batch
  losses = Plots.plot(0:n,
    duplast([it.memory.latest_batch.loss.L for it in iterations]),
    title="Loss on Last Batch",
    legend=:topright,
    ylims=(0, Inf),
    label="L")
  Plots.plot!(losses, 0:n,
    duplast([it.memory.latest_batch.loss.Lv for it in iterations]),
    label="Lv")
  Plots.plot!(losses, 0:n,
    duplast([it.memory.latest_batch.loss.Lp for it in iterations]),
    label="Lp")
  # Loss per game stage
  nstages = params.num_game_stages
  colors = range(colorant"blue", stop=colorant"red", length=nstages)
  pslosses = Plots.plot(title="Loss per Game Stage", ylims=(0, Inf))
  for s in 1:nstages
    Plots.plot!(pslosses, 0:n,
      duplast([
        it.memory.per_game_stage[s].samples_stats.loss.L
        for it in iterations]),
      label="$s",
      color=colors[s])
  end
  # Number of epochs
  nepochs = Plots.plot(
    duplast([length(it.learning.epochs) for it in iterations]),
    title="Number of learning epochs",
    legend=:false,
    t=:bar)
  # Policies entropy
  entropies = Plots.plot(0:n,
    duplast([it.memory.latest_batch.Hp for it in iterations]),
    ylims=(0, Inf),
    title="Policy Entropy",
    label="MCTS")
  Plots.plot!(entropies, 0:n,
    duplast([it.memory.latest_batch.Hp̂ for it in iterations]),
    label="Network")
  # Network statistics
  net = Plots.plot(0:n,
    duplast([it.learning.initial_status.network.maxw for it in iterations]),
    title="Neural Network Statistics",
    label="Max W")
  Plots.plot!(net, 0:n,
    duplast([it.learning.initial_status.network.meanw for it in iterations]),
    label="Mean W")
  Plots.plot!(net, 0:n,
    duplast([it.learning.initial_status.network.vbias for it in iterations]),
    label="Value bias")
  # Assembling everything together
  append!(plots, [
    arena, nepochs, pslosses, losses, entropies, net, nsamples])
  append!(files, [
    "arena", "nepochs", "loss_per_stage",
    "loss", "entropies", "net", "nsamples"])
  for (file, plot) in zip(files, plots)
    #Plots.plot!(plot, dpi=200, size=(600, 200))
    Plots.savefig(plot, joinpath(dir, file))
  end
end
