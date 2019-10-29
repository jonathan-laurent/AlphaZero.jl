#####
##### Loss Plot
#####

function plot_losses(getlosses, range, title)
  fields = fieldnames(Report.Loss)
  labels = [string(f) for f in fields]
  data = [[getfield(getlosses(i), f) for i in range] for f in fields]
  return Plots.plot(range, data,
    label=labels, title=title, ylims=(0, Inf))
end

#####
##### Iteration summary plots
#####

function plot_iteration(
    report::Report.Iteration,
    params::Params,
    dir::String)
  isdir(dir) || mkpath(dir)
  epochs = report.learning.epochs
  losses_plot = plot_losses(0:length(epochs), "Losses") do i
    if i == 0
      report.learning.initial_status.loss
    else
      epochs[i].status_after.loss
    end
  end
  checkpoints = report.learning.checkpoints
  checkpoints_plot = Plots.hline(
    [0, params.arena.update_threshold],
    title="Checkpoints")
  Plots.plot!(checkpoints_plot,
    [c.epoch_id for c in checkpoints],
    [c.reward for c in checkpoints],
    t=:scatter,
    legend=:none)
  plot = Plots.plot(losses_plot, checkpoints_plot, layout=(2, 1))
  Plots.savefig(plot, joinpath(dir, "summary"))
end

#####
##### Training summary plots
#####

function plot_training(
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
      ylims=(-1.0, 1.0),
      legend=nothing)
    Plots.hline!(vbars, [0])
    push!(plots, vbars)
    push!(files, "validation")
  end
  # Number of samples
  nsamples = Plots.plot(0:n,
    [0;[it.memory.all_samples.num_samples for it in iterations]],
    title="Experience Buffer Size",
    label="Number of samples")
  Plots.plot!(nsamples, 0:n,
    [0;[it.memory.all_samples.num_boards for it in iterations]],
    label="Number of distinct boards")
  # Performances during evaluation
  arena = Plots.plot(1:n, [
    maximum(c.reward for c in it.learning.checkpoints)
    for it in iterations],
    title="Arena Results",
    ylims=(-1, 1),
    t=:bar,
    legend=:none)
  Plots.hline!(arena, [0, params.arena.update_threshold])
  # Loss on last batch
  losses_last = plot_losses(1:n, "Loss on last batch") do i
    iterations[i].memory.latest_batch.status.loss
  end
  losses_fullmem = plot_losses(1:n, "Loss on full memory") do i
    iterations[i].memory.all_samples.status.loss
  end
  # Loss per game stage
  nstages = params.num_game_stages
  colors = range(colorant"blue", stop=colorant"red", length=nstages)
  pslosses = Plots.plot(title="Loss per Game Stage", ylims=(0, Inf))
  for s in 1:nstages
    Plots.plot!(pslosses, 1:n, [
        it.memory.per_game_stage[s].samples_stats.status.loss.L
        for it in iterations],
      label="$s",
      color=colors[s])
  end
  # Number of epochs
  nepochs = Plots.plot(1:n,
    [length(it.learning.epochs) for it in iterations],
    title="Number of learning epochs",
    legend=:false,
    t=:bar)
  # Policies entropy
  entropies = Plots.plot(1:n,
    [it.memory.latest_batch.Hp for it in iterations],
    ylims=(0, Inf),
    title="Policy Entropy",
    label="MCTS")
  Plots.plot!(entropies, 1:n,
    [it.memory.latest_batch.status.Hpnet for it in iterations],
    label="Network")
  # Performance reports
  perfs_global_labels = ["Self Play", "Memory Analysis", "Learning"]
  perfs_global_content = [
    sum(it.time_self_play for it in iterations),
    sum(it.time_memory_analysis for it in iterations),
    sum(it.time_learning for it in iterations)]
  if !isnothing(validation)
    push!(perfs_global_labels, "Validation")
    push!(perfs_global_content, sum(rep.time for rep in validation))
  end
  perfs_global = Plots.pie(
    perfs_global_labels, perfs_global_content, title="Global")
  perfs_self_play =
    let itratio = mean(it.self_play.inference_time_ratio for it in iterations)
      Plots.pie(
        ["Inference", "MCTS"], [itratio, 1-itratio],
        title="Self Play") end
  perfs_learning = Plots.pie(
    ["Samples Conversion", "Loss computation", "Optimization", "Evaluation"], [
      sum(it.learning.time_convert for it in iterations),
      sum(it.learning.time_loss for it in iterations),
      sum(it.learning.time_train for it in iterations),
      sum(it.learning.time_eval for it in iterations)],
    title="Learning")
  perfs = Plots.plot(
    perfs_global, perfs_self_play, perfs_learning)
  # Assembling everything together
  append!(plots, [
    arena, nepochs, pslosses, losses_fullmem, losses_last,
    entropies, nsamples, perfs])
  append!(files, [
    "arena", "nepochs", "loss_per_stage", "loss_fullmem", "loss_last_batch",
    "entropies", "nsamples", "perfs"])
  for (file, plot) in zip(files, plots)
    #Plots.plot!(plot, dpi=200, size=(600, 200))
    Plots.savefig(plot, joinpath(dir, file))
  end
end
