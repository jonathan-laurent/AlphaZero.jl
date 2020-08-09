#####
##### Session management
#####

import AlphaZero: Handlers, initial_report, get_experience

"""
    SessionReport

The full collection of statistics and benchmark results
collected during a training session.

# Fields
- `iterations`: vector of ``n`` iteration reports with type
    [`Report.Iteration`](@ref)
- `benchmark`: vector of ``n+1`` benchmark reports with type
    [`Benchmark.Report`](@ref)
"""
struct SessionReport
  iterations :: Vector{Report.Iteration}
  benchmark :: Vector{Benchmark.Report}
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
  dir :: Option{String}
  logger :: Logger
  autosave :: Bool
  save_intermediate :: Bool
  benchmark :: Vector{Benchmark.Duel}
  # Temporary state for logging
  progress :: Option{Progress}
  report :: SessionReport

  function Session(env, dir, logger, autosave, save_intermediate, benchmark)
    return new{typeof(env)}(
      env, dir, logger, autosave, save_intermediate,
      benchmark, nothing, SessionReport())
  end
end

AlphaZero.GameType(::Session{<:Env{G}}) where {G} = G

#####
##### Save and load environments
#####

const BESTNN_FILE      =  "bestnn.data"
const CURNN_FILE       =  "curnn.data"
const MEM_FILE         =  "mem.data"
const ITC_FILE         =  "iter.txt"
const REPORT_FILE      =  "report.json"
const PARAMS_FILE      =  "params.json"
const NET_PARAMS_FILE  =  "netparams.json"
const BENCHMARK_FILE   =  "benchmark.json"
const LOG_FILE         =  "log.txt"
const PLOTS_DIR        =  "plots"
const ITERS_DIR        =  "iterations"

iterdir(dir, i) = joinpath(dir, ITERS_DIR, "$i")

function valid_session_dir(dir)
  !isnothing(dir) &&
  isfile(joinpath(dir, PARAMS_FILE)) &&
  isfile(joinpath(dir, BESTNN_FILE)) &&
  isfile(joinpath(dir, CURNN_FILE)) &&
  isfile(joinpath(dir, MEM_FILE)) &&
  isfile(joinpath(dir, ITC_FILE))
end

function save_env(env::Env, dir)
  isdir(dir) || mkpath(dir)
  # Saving parameters
  open(joinpath(dir, PARAMS_FILE), "w") do io
    JSON2.pretty(io, JSON3.write(env.params))
  end
  open(joinpath(dir, NET_PARAMS_FILE), "w") do io
    JSON2.pretty(io, JSON3.write(Network.hyperparams(env.bestnn)))
  end
  # Saving state
  serialize(joinpath(dir, BESTNN_FILE), env.bestnn)
  serialize(joinpath(dir, CURNN_FILE), env.curnn)
  serialize(joinpath(dir, MEM_FILE), get_experience(env))
  open(joinpath(dir, ITC_FILE), "w") do io
    JSON3.write(io, env.itc)
  end
end

function load_network(logger, net_file, netparams_file)
  # Try to load network or otherwise network params
  if isfile(net_file)
    network = deserialize(net_file)
    Log.print(logger, "Loading network from: $(net_file)")
  else
    Log.print(logger, Log.RED, "No network file: $(net_file)")
    network = open(netparams_file, "r") do io
      params = JSON3.read(io, HyperParams(Network))
      Network(params)
    end
  end
  return network
end

function load_env(
    ::Type{Game}, ::Type{Network}, logger, dir; params
  ) where {Game, Network}
  Log.section(logger, 1, "Loading environment")
  # Load the neural networks
  netparams_file = joinpath(dir, NET_PARAMS_FILE)
  bestnn = load_network(logger, joinpath(dir, BESTNN_FILE), netparams_file)
  curnn = load_network(logger, joinpath(dir, CURNN_FILE), netparams_file)
  # Load memory
  mem_file = joinpath(dir, MEM_FILE)
  if isfile(mem_file)
    experience = deserialize(mem_file)
    Log.print(logger, "Loading memory from: $(mem_file)")
  else
    experience = []
    Log.print(logger, Log.RED, "Starting with an empty memory")
  end
  # Load instructions counter
  itc_file = joinpath(dir, ITC_FILE)
  if isfile(itc_file)
    itc = open(itc_file, "r") do io
      JSON3.read(io)
    end
    Log.print(logger, "Loaded iteration counter from: $(itc_file)")
  else
    itc = 0
    Log.print(logger, Log.RED, "File not found: $(itc_file)")
  end
  return Env{Game}(params, curnn, bestnn, experience, itc)
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
    isfile(bfile) || error("Not found: $bfile")
    open(bfile, "r") do io
      push!(rep.benchmark, JSON3.read(io, Benchmark.Report))
    end
    # Load the iteration report
    if itc > 0
      isfile(ifile) || error("Not found: $ifile")
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
    JSON2.pretty(io, JSON3.write(bench))
  end
  if session.env.itc > 0
    @assert !isnothing(itrep)
    open(joinpath(idir, REPORT_FILE), "w") do io
      JSON2.pretty(io, JSON3.write(itrep))
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

win_rate(z) = round(Int, 100 * (z + 1) / 2)

percentage(x, total) = round(Int, 100 * (x / total))

function show_space_after_progress_bar(logger)
  Log.console_only(logger) do
    Log.sep(logger, force=true)
  end
end

function run_duel(env, logger, duel)
  player_name = Benchmark.name(duel.player)
  baseline_name = Benchmark.name(duel.baseline)
  legend = "$player_name against $baseline_name"
  Log.section(logger, 2, "Running benchmark: $legend")
  progress = Log.Progress(logger, duel.num_games)
  outcome = Benchmark.run(env, duel, progress)
  show_space_after_progress_bar(logger)
  z = fmt("+.2f", outcome.avgr)
  if env.params.ternary_rewards
    stats = Benchmark.TernaryOutcomeStatistics(outcome)
    n = length(outcome.rewards)
    pwon = percentage(stats.num_won, n)
    pdraw = percentage(stats.num_draw, n)
    plost = percentage(stats.num_lost, n)
    details = "$pwon% won, $pdraw% draw, $plost% lost"
  else
    wr = win_rate(outcome.avgr)
    details = "win rate of $wr%"
  end
  red = fmt(".1f", 100 * outcome.redundancy)
  msg = "Average reward: $z ($details), redundancy: $red%"
  Log.print(logger, msg)
  return outcome
end

function run_benchmark(session)
  report = Benchmark.Report()
  for duel in session.benchmark
    outcome = run_duel(session.env, session.logger, duel)
    push!(report, outcome)
  end
  return report
end

function zeroth_iteration!(session::Session)
  @assert session.env.itc == 0
  Log.section(session.logger, 2, "Initial report")
  print_report(session.logger, initial_report(session.env))
  bench = run_benchmark(session)
  save_increment!(session, bench)
end

#####
##### Building the logger
#####

function session_logger(dir, nostdout, autosave)
  if !isnothing(dir) && autosave
    isdir(dir) || mkpath(dir)
    logfile = open(joinpath(dir, LOG_FILE), "a")
  else
    logfile = devnull
  end
  out = nostdout ? devnull : stdout
  return Logger(out, logfile=logfile)
end

#####
##### Session constructors
#####

"""
    Session(::Type{Game}, ::Type{Net}, params, netparams) where {Game, Net}

Create a new session using the given parameters, or load it from disk if
it already exists.

# Arguments
- `Game` is the type ot the game that is being learnt
- `Net` is the type of the network that is being used
- `params` has type [`Params`](@ref)
- `netparams` has type [`Network.HyperParams(Net)`](@ref Network.HyperParams)

# Optional keyword arguments
- `dir`: session directory in which all files and reports are saved; this
    argument is either a string or `nothing` (default), in which case the
    session won't be saved automatically and no file will be generated
- `autosave=true`: if set to `false`, the session won't be saved automatically nor
    any file will be generated
- `nostdout=false`: disables logging on the standard output when set to `true`
- `benchmark=[]`: vector of [`Benchmark.Duel`](@ref) to be used as a benchmark
- `save_intermediate=false`: if set to true (along with `autosave`), all
    intermediate training environments are saved on disk so that
    the whole training process can be analyzed later. This can
    consume a lot of disk space.
"""
function Session(
    ::Type{Game}, ::Type{Net}, params, netparams;
    dir=nothing, autosave=true, nostdout=false, benchmark=[],
    save_intermediate=false
  ) where {Game, Net}
  logger = session_logger(dir, nostdout, autosave)
  if valid_session_dir(dir)
    env = load_env(Game, Net, logger, dir, params=params)
    # The parameters must be unchanged
    same_json(x, y) = JSON3.write(x) == JSON3.write(y)
    same_json(env.params, params) || @info "Using modified parameters"
    @assert same_json(Network.hyperparams(env.bestnn), netparams)
    session = Session(env, dir, logger, autosave, save_intermediate, benchmark)
    session.report = load_session_report(dir, env.itc)
  else
    network = Net(netparams)
    env = Env{Game}(params, network)
    session = Session(env, dir, logger, autosave, save_intermediate, benchmark)
    Log.section(session.logger, 1, "Initializing a new AlphaZero environment")
    zeroth_iteration!(session)
  end
  return session
end

"""
    Session(env::Env[, dir])

Create a session from an initial environment.

- The iteration counter of the environment must be equal to 0
- If a session directory is provided, this directory must not exist yet

This constructor features the optional keyword arguments
`autosave`, `nostdout`, `benchmark` and `save_intermediate`.
"""
function Session(
    env::Env, dir=nothing; autosave=true, nostdout=false, benchmark=[],
    save_intermediate=false)
  @assert isnothing(dir) || !isdir(dir)
  @assert env.itc == 0
  logger = session_logger(dir, nostdout, autosave)
  session = Session(env, dir, logger, autosave, save_intermediate, benchmark)
  zeroth_iteration!(session)
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
    start_explorer(session::Session)

Start an explorer session for the current environment. See [`Explorer`](@ref).
"""
function start_explorer(session::Session; mcts_params=nothing, on_gpu=false)
  isnothing(mcts_params) && (mcts_params = session.env.params.self_play.mcts)
  Log.section(session.logger, 1, "Starting interactive exploration")
  explorer = Explorer(session.env, mcts_params=mcts_params, on_gpu=on_gpu)
  start_explorer(explorer)
end

"""
    play_interactive_game(session::Session; timeout=2.)

Start an interactive game against AlphaZero, allowing it
`timeout` seconds of thinking time for each move.
"""
function play_interactive_game(
    session::Session; timeout=2., mcts_params=nothing, on_gpu=false)
  Game = GameType(session)
  net = Network.copy(session.env.bestnn, on_gpu=on_gpu, test_mode=true)
  isnothing(mcts_params) && (mcts_params = session.env.params.self_play.mcts)
  player = MctsPlayer(net, mcts_params, timeout=timeout)
  interactive!(Game(), player, Human{Game}())
end

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
  Log.print(logger, "MCTS memory footprint: $(memf)B")
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
end

#####
##### Event handlers
#####

function Handlers.iteration_started(session::Session)
  i = session.env.itc + 1
  Log.section(session.logger, 1, "Starting iteration $i")
end

function Handlers.self_play_started(session::Session)
  ngames = session.env.params.self_play.num_games
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

function Handlers.learning_started(session::Session, initial_status)
  Log.section(session.logger, 2, "Starting learning")
  Log.section(session.logger, 3, "Optimizing the loss")
  print_report(session.logger, initial_status, style=Log.BOLD)
end

function Handlers.updates_finished(session::Session, report)
  print_report(session.logger, report)
end

function Handlers.checkpoint_started(session::Session)
  Log.section(session.logger, 3, "Launching a checkpoint evaluation")
  num_games = session.env.params.arena.num_games
  session.progress = Log.Progress(session.logger, num_games)
end

function Handlers.checkpoint_game_played(session::Session)
  next!(session.progress)
end

function Handlers.checkpoint_finished(session::Session, report)
  show_space_after_progress_bar(session.logger)
  avgr = fmt("+.2f", report.reward)
  wr = win_rate(report.reward)
  red = fmt(".1f", report.redundancy * 100)
  nnr = report.nn_replaced ? ", network replaced" : ""
  msg = "Average reward: $avgr (win rate of $wr%$nnr), redundancy: $red%"
  Log.print(session.logger, msg)
  Log.section(session.logger, 3, "Optimizing the loss")
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

function run_duel(
    ::Type{G}, ::Type{N}, dir::String, duel::Benchmark.Duel;
    params=nothing) where {G, N}
  logger = Logger()
  env = load_env(G, N, logger, dir, params=params)
  run_duel(env, logger, duel)
end

function run_new_benchmark(
  ::Type{G}, ::Type{N}, session_dir, name, benchmark;
  params=nothing, itcmax=nothing) where {G, N}
  outdir = joinpath(session_dir, name)
  isdir(outdir) || mkpath(outdir)
  logger = Logger()
  Log.section(logger, 1, "Computing new benchmark: $name")
  itc = 0
  reports = Benchmark.Report[]
  while valid_session_dir(iterdir(session_dir, itc))
    !isnothing(itcmax) && itc > itcmax && break
    Log.section(logger, 1, "Iteration: $itc")
    itdir = iterdir(session_dir, itc)
    env = load_env(G, N, logger, itdir, params=params)
    report = [run_duel(env, logger, duel) for duel in benchmark]
    push!(reports, report)
    # Save the intermediate reports
    open(joinpath(outdir, BENCHMARK_FILE), "w") do io
      JSON2.pretty(io, JSON3.write(reports))
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
