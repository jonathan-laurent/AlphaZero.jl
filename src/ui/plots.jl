#####
##### Loss Plot
#####

function plot_losses(getlosses, range, title, xlabel=:none)
  fields = fieldnames(Report.Loss)
  labels = [string(f) for _ in 1:1, f in fields]
  data = [(getlosses(i)..., f) for i in range, f in fields]
  xs = map(d -> d[1], data)
  ys = map(d -> getfield(d[2], d[3]), data)
  xlims = (minimum(xs), maximum(xs))
  return Plots.plot(xs, ys,
    label=labels, title=title, xlims=xlims, ylims=(0, Inf), xlabel=xlabel)
end

#####
##### Iteration summary plots
#####

function learning_iter_plot(rep::Report.Learning, params::Params)
  n = length(rep.checkpoints)
  if n == 0
    return nothing
  else
    nbatches = rep.checkpoints[end].batch_id
    losses_plot = plot_losses(0:n, "Losses") do i
      if i == 0
        (0, rep.initial_status.loss)
      else
        (rep.checkpoints[i].batch_id, rep.checkpoints[i].status_after.loss)
      end
    end
    checkpoints_plot = Plots.hline(
      [0, params.arena.update_threshold],
      title="Checkpoints")
    Plots.plot!(checkpoints_plot,
      [c.batch_id for c in rep.checkpoints],
      [c.evaluation.avgr for c in rep.checkpoints],
      ylims=(-1.0, 1.0),
      t=:scatter,
      legend=:none)
    Plots.xlims!(losses_plot, (0, nbatches))
    Plots.xlims!(checkpoints_plot, (0, nbatches))
    return Plots.plot(losses_plot, checkpoints_plot, layout=(2, 1))
  end
end

function performances_plot(rep::Report.Iteration)
  # Global
  global_labels = []
  global_content = []
  push!(global_labels, "Self Play")
  push!(global_content, rep.perfs_self_play.time)
  if !isnothing(rep.memory)
    push!(global_labels, "Memory Analysis")
    push!(global_content, rep.perfs_memory_analysis.time)
  end
  push!(global_labels, "Learning")
  push!(global_content, rep.perfs_learning.time)
  glob = Plots.pie(global_labels, global_content,
    title="Global",
    legend=:right)
  # Self-play details
  self_play =
    let gcratio = rep.perfs_self_play.gc_time / rep.perfs_self_play.time
      Plots.pie(
        ["MCTS and Inference", "GC"],
        [1 - gcratio, gcratio],
        title="Self Play")
    end
  # Learning details
  learning = Plots.pie(
    ["Samples conversion", "Loss computation", "Optimization", "Arena (MCTS)"],
    [ rep.learning.time_convert,
      rep.learning.time_loss,
      rep.learning.time_train,
      rep.learning.time_eval],
    title="Learning")
  return Plots.plot(glob, self_play, learning)
end

function plot_iteration(
    report::Report.Iteration,
    params::Params,
    dir::String,
    itc::Int)
  plots = []
  # Summary plot
  splot = learning_iter_plot(report.learning, params)
  isnothing(splot) || push!(plots, ("iter_summary", splot))
  # Performances plot
  pplot = performances_plot(report)
  push!(plots, ("iter_perfs", pplot))
  # Losses plot
  losses = Util.momentum_smoothing(report.learning.losses, 0.1)
  lplot = Plots.plot(collect(eachindex(losses)), losses,
    title="Loss on Minibatches",
    ylims=(0, Inf),
    legend=nothing,
    xlabel="Batch number")
  push!(plots, ("iter_loss", lplot))
  # Saving everything
  for (name, plot) in plots
    pdir = joinpath(dir, name)
    isdir(pdir) || mkpath(pdir)
    Plots.savefig(plot, joinpath(pdir, "$itc"))
  end
end

#####
##### Training summary plots
#####

function plot_benchmark(
    params::Params,
    benchs::Vector{Report.Benchmark},
    dir::String)
  isempty(benchs) && return
  n = length(benchs) - 1
  nduels = length(benchs[1])
  nduels >= 1 || return
  @assert all(length(b) == nduels for b in benchs)
  isdir(dir) || mkpath(dir)
  labels = [b.legend for _ in 1:1, b in benchs[1]]
  # Average reward
  avgr_data = [[b[i].avgr for b in benchs] for i in 1:nduels]
  avgr = Plots.plot(0:n,
    avgr_data,
    title="Average Reward",
    ylims=(params.ternary_rewards ? (-1.0, 1.0) : nothing),
    legend=:bottomright,
    label=labels,
    xlabel="Iteration number")
  Plots.savefig(avgr, joinpath(dir, "benchmark_reward"))
  if params.ternary_rewards
    function compute_percentage(b, f)
      stats = Benchmark.TernaryOutcomeStatistics(b)
      return ceil(Int, 100 * (f(stats) / length(b.rewards)))
    end
    function compute_data(f)
      return [[compute_percentage(b[i], f) for b in benchs] for i in 1:nduels]
    end
    # Percentage of lost games
    ploss = Plots.plot(0:n,
      compute_data(s -> s.num_lost),
      title="Percentage of Lost Games",
      ylims=(0, 100),
      legend=:topright,
      label=labels,
      xlabel="Iteration number")
    Plots.savefig(ploss, joinpath(dir, "benchmark_lost_games"))
    # Percentage of won games
    pwin = Plots.plot(0:n,
      compute_data(s -> s.num_won),
      title="Percentage of Won Games",
      ylims=(0, 100),
      legend=:bottomright,
      label=labels,
      xlabel="Iteration number")
    Plots.savefig(pwin, joinpath(dir, "benchmark_won_games"))
  end
end

function plot_training(
    params::Params,
    iterations::Vector{Report.Iteration},
    dir::String)
  n = length(iterations)
  iszero(n) && return
  isdir(dir) || mkpath(dir)
  plots, files = [], []
  # Exploration depth
  expdepth = Plots.plot(1:n,
    [it.self_play.average_exploration_depth for it in iterations],
    title="Average Exploration Depth",
    ylims=(0, Inf),
    legend=:none,
    xlabel="Iteration number")
  # Number of samples
  nsamples = Plots.plot(0:n,
    [0;[it.self_play.memory_size for it in iterations]],
    title="Experience Buffer Size",
    label="Number of samples",
    xlabel="Iteration number")
  Plots.plot!(nsamples, 0:n,
    [0;[it.self_play.memory_num_distinct_boards for it in iterations]],
    label="Number of distinct boards")
  # Performances during evaluation
  if all(it -> !isempty(it.learning.checkpoints), iterations)
    arena = Plots.plot(1:n, [
      maximum(c.evaluation.avgr for c in it.learning.checkpoints)
      for it in iterations],
      title="Arena Results",
      ylims=(-1, 1),
      t=:bar,
      legend=:none,
      xlabel="Iteration number")
    Plots.hline!(arena, [0, params.arena.update_threshold])
    push!(plots, arena)
    push!(files, "arena")
  end
  # Loss on the full memory after an iteration
  lfmt = "Loss on Full Memory"
  losses_fullmem = plot_losses(1:n, lfmt, "Iteration number") do i
    (i, iterations[i].learning.initial_status.loss)
  end
  # Plots related to the memory analysis
  if all(it -> !isnothing(it.memory), iterations)
    # Loss on last batch
    losses_last = plot_losses(
      1:n, "Loss on Last Batch", "Iteration number") do i
      (i, iterations[i].memory.latest_batch.status.loss)
    end
    # Loss per game stage
    nstages = minimum(length(it.memory.per_game_stage) for it in iterations)
    colors = range(colorant"blue", stop=colorant"red", length=nstages)
    losses_ps = Plots.plot(
      title="Loss per Game Stage", ylims=(0, Inf), xlabel="Iteration number")
    for s in 1:nstages
      tmin = minimum([
        it.memory.per_game_stage[s].min_remaining_length
        for it in iterations])
      tmax = maximum([
        it.memory.per_game_stage[s].max_remaining_length
        for it in iterations])
      Plots.plot!(losses_ps, 1:n, [
          it.memory.per_game_stage[s].samples_stats.status.loss.L
          for it in iterations],
        label="$tmin to $tmax turns left",
        color=colors[s])
    end
    append!(plots, [losses_last, losses_ps])
    append!(files, ["loss_last_batch", "loss_per_stage"])
  end
  # Policies entropy
  entropies = Plots.plot(1:n,
    [it.learning.initial_status.Hp for it in iterations],
    ylims=(0, Inf),
    title="Policy Entropy",
    label="MCTS",
    xlabel="Iteration number")
  Plots.plot!(entropies, 1:n,
    [it.learning.initial_status.Hpnet for it in iterations],
    label="Network")
  # Assembling everything together
  append!(plots,
    [entropies, nsamples, expdepth, losses_fullmem])
  append!(files,
    ["entropies", "nsamples", "exploration_depth", "loss"])
  for (file, plot) in zip(files, plots)
    Plots.savefig(plot, joinpath(dir, file))
  end
end
# To test:
# AlphaZero.plot_training("sessions/connect-four")
