#####
##### Session management
#####

const DEFAULT_SESSIONS_DIR = "sessions"

"""
    SessionReport

The full collection of statistics and benchmark results
collected during a training session.

# Fields
- `iterations`: vector of ``n`` iteration reports with type
    [`Report.Iteration`](@ref)
- `benchmark`: vector of ``n+1`` benchmark reports with type
    [`Report.Benchmark`](@ref)
"""
struct SessionReport
  iterations :: Vector{Report.Iteration}
  benchmark :: Vector{Report.Benchmark}
  SessionReport() = new([], [])
end

function valid_session_report(r::SessionReport)
  if length(r.benchmark) == length(r.iterations) + 1
    nduels = length(r.benchmark[1])
    return all(length(b) == nduels for b in r.benchmark)
  else
    return false
  end
end

"""
    Session{Env}

A wrapper on an AlphaZero environment that adds features such as:
- Logging and plotting
- Loading and saving of environments
In particular, it implements the [`Handlers`](@ref) interface.

# Public fields
- `env::Env` is the environment wrapped by the session
- `report` is the current session report, with type [`SessionReport`](@ref)
"""
mutable struct Session{Env}
  env :: Env
  dir :: String
  logger :: Logger
  autosave :: Bool
  save_intermediate :: Bool
  benchmark :: Vector{Benchmark.Evaluation}
  # Temporary state for logging
  progress :: Union{Progress, Nothing}
  report :: SessionReport

  function Session(env, dir, logger, autosave, save_intermediate, benchmark)
    return new{typeof(env)}(
      env, dir, logger, autosave, save_intermediate,
      benchmark, nothing, SessionReport())
  end
end

#####
##### Save and load environments
#####

const GSPEC_FILE       =  "gspec.data"
const BESTNN_FILE      =  "bestnn.data"
const CURNN_FILE       =  "curnn.data"
const MEM_FILE         =  "mem.data"
const ITC_FILE         =  "iter.txt"
const REPORT_FILE      =  "report.json"
const PARAMS_FILE      =  "params.data"
const PARAMS_JSON_FILE =  "params.json"      # not used when loading envs
const NET_PARAMS_FILE  =  "netparams.json"
const BENCHMARK_FILE   =  "benchmark.json"
const LOG_FILE         =  "log.txt"
const PLOTS_DIR        =  "plots"
const ITERS_DIR        =  "iterations"

iterdir(dir, i) = joinpath(dir, ITERS_DIR, "$i")

function valid_session_dir(dir)
  isfile(joinpath(dir, PARAMS_FILE)) &&
  isfile(joinpath(dir, BESTNN_FILE)) &&
  isfile(joinpath(dir, CURNN_FILE)) &&
  isfile(joinpath(dir, MEM_FILE)) &&
  isfile(joinpath(dir, ITC_FILE))
end

function save_env(env::Env, dir)
  isdir(dir) || mkpath(dir)
  serialize(joinpath(dir, GSPEC_FILE), env.gspec)
  serialize(joinpath(dir, PARAMS_FILE), env.params)
  open(joinpath(dir, PARAMS_JSON_FILE), "w") do io
    JSON3.pretty(io, JSON3.write(env.params))
  end
  open(joinpath(dir, NET_PARAMS_FILE), "w") do io
    JSON3.pretty(io, JSON3.write(Network.hyperparams(env.bestnn)))
  end
  serialize(joinpath(dir, BESTNN_FILE), env.bestnn)
  serialize(joinpath(dir, CURNN_FILE), env.curnn)
  serialize(joinpath(dir, MEM_FILE), get_experience(env))
  open(joinpath(dir, ITC_FILE), "w") do io
    JSON3.write(io, env.itc)
  end
end

function load_env(dir)
  gspec = deserialize(joinpath(dir, GSPEC_FILE))
  params = deserialize(joinpath(dir, PARAMS_FILE))
  curnn = deserialize(joinpath(dir, CURNN_FILE))
  bestnn = deserialize(joinpath(dir, BESTNN_FILE))
  experience = deserialize(joinpath(dir, MEM_FILE))
  itc = open(JSON3.read, joinpath(dir, ITC_FILE), "r") 
  return Env(gspec, params, curnn, bestnn, experience, itc)
end

#####
##### Save and load session reports
#####

function load_session_report(dir::String, niters)
  rep = SessionReport()
  for itc in 0:niters
    idir = iterdir(dir, itc)
    ifile = joinpath(idir, REPORT_FILE)
    bfile = joinpath(idir, BENCHMARK_FILE)
    # Load the benchmark report
    open(bfile, "r") do io
      push!(rep.benchmark, JSON3.read(io, Report.Benchmark))
    end
    # Load the iteration report
    if itc > 0
      open(ifile, "r") do io
        push!(rep.iterations, JSON3.read(io, Report.Iteration))
      end
    end
  end
  @assert valid_session_report(rep)
  return rep
end

function save_report_increment(session, bench, itrep, idir)
  open(joinpath(idir, BENCHMARK_FILE), "w") do io
    JSON3.pretty(io, JSON3.write(bench))
  end
  if session.env.itc > 0
    @assert !isnothing(itrep)
    open(joinpath(idir, REPORT_FILE), "w") do io
      JSON3.pretty(io, JSON3.write(itrep))
    end
  end
end

#####
##### Incremental saving
#####

autosave_enabled(s::Session) = s.autosave && !isnothing(s.dir)

function save_increment!(session::Session, bench, itrep=nothing)
  push!(session.report.benchmark, bench)
  isnothing(itrep) || push!(session.report.iterations, itrep)
  itc = session.env.itc
  if autosave_enabled(session)
    idir = iterdir(session.dir, itc)
    isdir(idir) || mkpath(idir)
    # Save the environment state,
    # both at the root and in the last iteration folder
    save(session, session.dir)
    session.save_intermediate && save(session, idir)
    # Save the collected statistics
    save_report_increment(session, bench, itrep, idir)
    # Do the plotting
    params = session.env.params
    plotsdir = joinpath(session.dir, PLOTS_DIR)
    isnothing(itrep) || plot_iteration(itrep, params, plotsdir, itc)
    plot_training(params, session.report.iterations, plotsdir)
    plot_benchmark(params, session.report.benchmark, plotsdir)
  end
end

#####
##### Run benchmarks
#####

function show_space_after_progress_bar(logger)
  Log.console_only(logger) do
    Log.sep(logger, force=true)
  end
end

function run_duel(env::Env, duel; logger)
  if isa(duel, Benchmark.Duel)
    player_name = Benchmark.name(duel.player)
    baseline_name = Benchmark.name(duel.baseline)
    legend = "$player_name against $baseline_name"
  else
    @assert isa(duel, Benchmark.Single)
    legend = Benchmark.name(duel.player)
  end
  Log.section(logger, 2, "Running benchmark: $legend")
  progress = Log.Progress(logger, duel.sim.num_games)
  report = Benchmark.run(env, duel, progress)
  show_space_after_progress_bar(logger)
  print_report(
    logger, report,
    ternary_rewards=env.params.ternary_rewards)
  return report
end

function run_benchmark(session::Session)
  report = Report.Benchmark()
  for duel in session.benchmark
    outcome = run_duel(session.env, duel, logger=session.logger)
    push!(report, outcome)
  end
  return report
end

# return whether or not critical problems were found
function zeroth_iteration!(session::Session)
  @assert session.env.itc == 0
  Log.section(session.logger, 2, "Initial report")
  report = initial_report(session.env)
  print_report(session.logger, report)
  isempty(report.errors) || return false
  bench = run_benchmark(session)
  save_increment!(session, bench)
  return true
end

function missing_zeroth_iteration(session::Session)
  return isempty(session.report.iterations) && isempty(session.report.benchmark)
end

#####
##### Session constructors
#####

default_session_dir(e::Experiment) = default_session_dir(e.name)
default_session_dir(exp_name::String) = joinpath(DEFAULT_SESSIONS_DIR, exp_name)

function session_logger(dir, nostdout, autosave)
  if autosave
    isdir(dir) || mkpath(dir)
    logfile = open(joinpath(dir, LOG_FILE), "a")
  else
    logfile = devnull
  end
  out = nostdout ? devnull : stdout
  return Logger(out, logfile=logfile)
end

"""
    Session(::Experiment; <optional kwargs>)

Create a new session from an experiment.

# Optional keyword arguments
- `dir="sessions/<experiment-name>"`: session directory in which all files and reports
    are saved.
- `autosave=true`: if set to `false`, the session won't be saved automatically nor
    any file will be generated
- `nostdout=false`: disables logging on the standard output when set to `true`
- `save_intermediate=false`: if set to true (along with `autosave`), all
    intermediate training environments are saved on disk so that
    the whole training process can be analyzed later. This can
    consume a lot of disk space.
"""
function Session(
    e::Experiment;
    dir=nothing,
    autosave=true,
    nostdout=false,
    save_intermediate=false)
  
  isnothing(dir) && (dir = default_session_dir(e))
  logger = session_logger(dir, nostdout, autosave)
  if valid_session_dir(dir)
    Log.section(logger, 1, "Loading environment from: $dir")
    env = load_env(dir)
    # The parameters must be unchanged
    same_json(x, y) = JSON3.write(x) == JSON3.write(y)
    same_json(env.params, e.params) || @info "Using modified parameters"
    @assert same_json(Network.hyperparams(env.bestnn), e.netparams)
    session = Session(env, dir, logger, autosave, save_intermediate, e.benchmark)
    session.report = load_session_report(dir, env.itc)
  else
    network = e.mknet(e.gspec, e.netparams)
    env = Env(e.gspec, e.params, network)
    session = Session(env, dir, logger, autosave, save_intermediate, e.benchmark)
    Log.section(session.logger, 1, "Initializing a new AlphaZero environment")
  end
  return session
end

#####
##### Public interface
#####

"""
    resume!(session::Session)

Resume a previously created or loaded session. The user can interrupt training
by sending a SIGKILL signal.
"""
function resume!(session::Session)
  try
    if missing_zeroth_iteration(session)
      success = zeroth_iteration!(session)
      success || return
    end
    train!(session.env, session)
  catch e
    isa(e, InterruptException) || rethrow(e)
    Log.section(session.logger, 1, "Interrupted by the user")
  end
end

"""
    save(session::Session)

Save a session on disk.

This function is called automatically by [`resume!`](@ref) after each
training iteration if the session was created with `autosave=true`.
"""
function save(session::Session, dir=session.dir)
  save_env(session.env, dir)
end

"""
    explore(session::Session, [mcts_params, use_gpu])

Start an explorer session for the current environment.
"""
function explore(session::Session; args...)
  Log.section(session.logger, 1, "Starting interactive exploration")
  explore(AlphaZeroPlayer(session.env; args...), session.env.gspec)
end

AlphaZero.AlphaZeroPlayer(s::Session; args...) = AlphaZero.AlphaZeroPlayer(s.env; args...)

#####
##### Utilities for printing reports
#####

const NUM_COL = Log.ColType(7, x -> fmt(".4f", x))
const BIGINT_COL = Log.ColType(10, n -> format(ceil(Int, n), commas=true))

const LEARNING_STATUS_TABLE = Log.Table([
  ("Loss",   NUM_COL,     s -> s.loss.L),
  ("Lv",     NUM_COL,     s -> s.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.loss.Linv),
  ("Hp",     NUM_COL,     s -> s.Hp),
  ("Hpnet",  NUM_COL,     s -> s.Hpnet)])

const SAMPLES_STATS_TABLE = Log.Table([
  ("Loss",   NUM_COL,     s -> s.status.loss.L),
  ("Lv",     NUM_COL,     s -> s.status.loss.Lv),
  ("Lp",     NUM_COL,     s -> s.status.loss.Lp),
  ("Lreg",   NUM_COL,     s -> s.status.loss.Lreg),
  ("Linv",   NUM_COL,     s -> s.status.loss.Linv),
  ("Hpnet",  NUM_COL,     s -> s.status.Hpnet),
  ("Hp",     NUM_COL,     s -> s.status.Hp),
  ("Wtot",   BIGINT_COL,  s -> s.Wtot),
  ("Nb",     BIGINT_COL,  s -> s.num_boards),
  ("Ns",     BIGINT_COL,  s -> s.num_samples)])

function print_report(logger::Logger, status::Report.LearningStatus; kw...)
  Log.table_row(logger, LEARNING_STATUS_TABLE, status; kw...)
end

function print_report(logger::Logger, stats::Report.Memory)
  content, styles, comments = [], [], []
  # All samples
  push!(content, stats.all_samples)
  push!(styles, Log.BOLD)
  push!(comments, ["all samples"])
  # Latest batch
  push!(content, stats.latest_batch)
  push!(styles, Log.BOLD)
  push!(comments, ["latest batch"])
  # Per game stage
  for stage in stats.per_game_stage
    minrem = stage.min_remaining_length
    maxrem = stage.max_remaining_length
    push!(content, stage.samples_stats)
    push!(styles, Log.NO_STYLE)
    push!(comments, ["$minrem to $maxrem turns left"])
  end
  Log.table(
    logger, SAMPLES_STATS_TABLE, content, styles=styles, comments=comments)
end

function print_report(logger::Logger, report::Report.SelfPlay)
  sspeed = format(round(Int, report.samples_gen_speed), commas=true)
  Log.print(logger, "Generating $(sspeed) samples per second on average")
  avgdepth = fmt(".1f", report.average_exploration_depth)
  Log.print(logger, "Average exploration depth: $avgdepth")
  memf = format(report.mcts_memory_footprint, autoscale=:metric, precision=2)
  Log.print(logger, "MCTS memory footprint per worker: $(memf)B")
  mems = format(report.memory_size, commas=true)
  memd = format(report.memory_num_distinct_boards, commas=true)
  Log.print(logger, "Experience buffer size: $(mems) ($(memd) distinct boards)")
end

function print_report(logger::Logger, report::Report.Initial)
  nnparams = format(report.num_network_parameters, commas=true)
  Log.print(logger, "Number of network parameters: $nnparams")
  nnregparams = format(report.num_network_regularized_parameters, commas=true)
  Log.print(logger, "Number of regularized network parameters: $nnregparams")
  mfpn = report.mcts_footprint_per_node
  Log.print(logger, "Memory footprint per MCTS node: $(mfpn) bytes")
  for w in report.warnings
    Log.print(logger, crayon"yellow", "Warning: ", w)
  end
  for e in report.errors
    Log.print(logger, crayon"red", "Error: ", e)
  end
end

percentage(x, total) = round(Int, 100 * (x / total))

function print_report(
    logger::Logger,
    report::Report.Evaluation;
    nn_replaced=false,
    ternary_rewards=false)

  r = fmt("+.2f", report.avgr)
  if ternary_rewards
    n = length(report.rewards)
    stats = Benchmark.TernaryOutcomeStatistics(report)
    pwon = percentage(stats.num_won, n)
    pdraw = percentage(stats.num_draw, n)
    plost = percentage(stats.num_lost, n)
    details = ["$pwon% won, $pdraw% draw, $plost% lost"]
  else
    wr = round(Int, 100 * (report.avgr + 1) / 2)
    details = []
  end
  if nn_replaced
     push!(details, "network replaced")
  end
  details = isempty(details) ? "" : " (" * join(details, ", ") * ")"
  red = fmt(".1f", 100 * report.redundancy)
  msg = "Average reward: $r$details, redundancy: $red%"
  Log.print(logger, msg)
end

function print_report(logger::Logger, report::Report.Checkpoint; ternary_rewards=false)
  print_report(
    logger, report.evaluation,
    nn_replaced=report.nn_replaced,
    ternary_rewards=ternary_rewards)
end

#####
##### Event handlers
#####

function Handlers.iteration_started(session::Session)
  i = session.env.itc + 1
  Log.section(session.logger, 1, "Starting iteration $i")
end

function Handlers.self_play_started(session::Session)
  ngames = session.env.params.self_play.sim.num_games
  Log.section(session.logger, 2, "Starting self-play")
  session.progress = Log.Progress(session.logger, ngames)
end

function Handlers.game_played(session::Session)
  next!(session.progress)
end

function Handlers.self_play_finished(session::Session, report)
  show_space_after_progress_bar(session.logger)
  print_report(session.logger, report)
  session.progress = nothing
end

function Handlers.memory_analyzed(session::Session, report)
  Log.section(session.logger, 2, "Memory Analysis")
  print_report(session.logger, report)
end

function Handlers.learning_started(session::Session)
  Log.section(session.logger, 2, "Starting learning")
end

function Handlers.updates_started(session::Session, status)
  Log.section(session.logger, 3, "Optimizing the loss")
  print_report(session.logger, status, style=Log.BOLD)
end

function Handlers.updates_finished(session::Session, report)
  print_report(session.logger, report)
end

function Handlers.checkpoint_started(session::Session)
  Log.section(session.logger, 3, "Launching a checkpoint evaluation")
  num_games = session.env.params.arena.sim.num_games
  # In single player games, each game has to be played twice (with both networks)
  n = GI.two_players(session.env.gspec) ? num_games : 2 * num_games
  session.progress = Log.Progress(session.logger, n)
end

function Handlers.checkpoint_game_played(session::Session)
  next!(session.progress)
end

function Handlers.checkpoint_finished(session::Session, report)
  show_space_after_progress_bar(session.logger)
  ternary_rewards = session.env.params.ternary_rewards
  print_report(session.logger, report, ternary_rewards=ternary_rewards)
end

function Handlers.learning_finished(session::Session, report)
  return
end

function Handlers.iteration_finished(session::Session, report)
  bench = run_benchmark(session)
  save_increment!(session, bench, report)
  flush(Log.logfile(session.logger))
end

function Handlers.training_finished(session::Session)
  Log.section(session.logger, 1, "Training completed")
  close(Log.logfile(session.logger))
end

#####
##### Launch new experiments on an existing session
#####

function run_duel(session_dir::String, duel; logger)
  env = load_env(session_dir)
  run_duel(env, duel; logger)
end

function run_new_benchmark(session_dir, name, benchmark; logger, itcmax=nothing)
  outdir = joinpath(session_dir, name)
  isdir(outdir) || mkpath(outdir)
  logger = Logger()
  Log.section(logger, 1, "Computing benchmark")
  itc = 0
  reports = Report.Benchmark[]
  while valid_session_dir(iterdir(session_dir, itc))
    !isnothing(itcmax) && itc > itcmax && break
    Log.section(logger, 1, "Iteration: $itc")
    itdir = iterdir(session_dir, itc)
    env = load_env(itdir)
    report = [run_duel(env, duel; logger) for duel in benchmark]
    push!(reports, report)
    # Save the intermediate reports
    open(joinpath(outdir, BENCHMARK_FILE), "w") do io
      JSON3.pretty(io, JSON3.write(reports))
    end
    plot_benchmark(env.params, reports, outdir)
    itc += 1
  end
end

function regenerate_plots(session::Session)
  plotsdir = joinpath(session.dir, PLOTS_DIR)
  plot_training(session.env.params, session.report.iterations, plotsdir)
  plot_benchmark(session.env.params, session.report.benchmark, plotsdir)
  for (itc, itrep) in enumerate(session.report.iterations)
    plot_iteration(itrep, session.env.params, plotsdir, itc)
  end
end
