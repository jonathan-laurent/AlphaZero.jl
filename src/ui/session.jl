#####
##### Session management
#####

"""
    SessionReport

The full collection of benchmarks and statistics
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
  benchmark :: Vector{Benchmark.Duel}
  # Temporary state for logging
  progress :: Option{Progress}
  report :: SessionReport

  function Session(env, dir, logger, autosave, benchmark)
    return new{typeof(env)}(
      env, dir, logger, autosave, benchmark, nothing, SessionReport())
  end
end

#####
##### Save and load environments
#####

const NET_FILE         =  "net.data"
const MEM_FILE         =  "mem.data"
const ITC_FILE         =  "iter.txt"
const REPORT_FILE      =  "report.json"
const PARAMS_FILE      =  "params.json"
const NET_PARAMS_FILE  =  "netparams.json"
const BENCHMARK_FILE   =  "benchmark.json"
const LOG_FILE         =  "log.txt"
const PLOTS_DIR        =  "plots"

iterdir(dir, i) = joinpath(dir, "$i")

function valid_session_dir(dir)
  !isnothing(dir) &&
  isfile(joinpath(dir, PARAMS_FILE)) &&
  isfile(joinpath(dir, NET_FILE)) &&
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
  serialize(joinpath(dir, NET_FILE), env.bestnn)
  serialize(joinpath(dir, MEM_FILE), get(env.memory))
  open(joinpath(dir, ITC_FILE), "w") do io
    JSON3.write(io, env.itc)
  end
end

function load_env(
    ::Type{Game}, ::Type{Network}, logger, dir; params=nothing
  ) where {Game, Network}
  Log.section(logger, 1, "Loading environment")
  # Load parameters
  if isnothing(params)
    params_file = joinpath(dir, PARAMS_FILE)
    params = open(params_file, "r") do io
      JSON3.read(io, Params)
    end
    Log.print(logger, "Loading parameters from: $(params_file)")
  end
  # Try to load network or otherwise network params
  net_file = joinpath(dir, NET_FILE)
  netparams_file = joinpath(dir, NET_PARAMS_FILE)
  if isfile(net_file)
    network = deserialize(net_file)
    Log.print(logger, "Loading network from: $(net_file)")
  else
    network = open(netparams_file, "r") do io
      params = JSON3.read(io, HyperParams(Network))
      Network(params)
    end
  end
  # Load memory
  mem_file = joinpath(dir, MEM_FILE)
  if isfile(mem_file)
    experience = deserialize(mem_file)
    Log.print(logger, "Loading memory from: $(mem_file)")
  else
    experience = []
    Log.print(logger, Crayon.RED, "Starting with an empty memory")
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
    Log.print(logger, Crayon.RED, "File not found: $(itc_file)")
  end
  return Env{Game}(params, network, experience, itc)
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
  if autosave_enabled(session)
    idir = iterdir(session.dir, session.env.itc)
    isdir(idir) || mkpath(idir)
    # Save the environment state,
    # both at the root and in the last iteration folder
    save(session, session.dir)
    save(session, idir)
    # Save the collected statistics
    save_report_increment(session, bench, itrep, idir)
    # Do the plotting
    params = session.env.params
    isnothing(itrep) || plot_iteration(itrep, params, idir)
    plotsdir = joinpath(session.dir, PLOTS_DIR)
    plot_training(params, session.report.iterations, plotsdir)
    plot_benchmark(params, session.report.benchmark, plotsdir)
  end
end

#####
##### Run benchmarks
#####

win_rate(z) = round(Int, 100 * (z + 1) / 2)

percentage(x, total) = round(Int, 100 * (x / total))

function show_space_after_progress_bar(session)
  Log.console_only(session.logger) do
    Log.sep(session.logger, force=true)
  end
end

function run_benchmark(session)
  report = Benchmark.Report()
  for duel in session.benchmark
    player_name = Benchmark.name(duel.player)
    baseline_name = Benchmark.name(duel.baseline)
    legend = "$player_name against $baseline_name"
    Log.section(session.logger, 2, "Running benchmark: $legend")
    progress = Log.Progress(session.logger, duel.num_games)
    outcome = Benchmark.run(session.env, duel, progress)
    push!(report, outcome)
    show_space_after_progress_bar(session)
    z = fmt("+.2f", outcome.avgz)
    if session.env.params.ternary_rewards
      stats = Benchmark.TernaryOutcomeStatistics(outcome)
      n = length(outcome.rewards)
      pwon = percentage(stats.num_won, n)
      pdraw = percentage(stats.num_draw, n)
      plost = percentage(stats.num_lost, n)
      details = "$pwon% won, $pdraw% draw, $plost% lost"
    else
      wr = win_rate(outcome.avgz)
      details = "win rate of $wr%"
    end
    red = fmt(".1f", 100 * outcome.redundancy)
    msg = "Average reward: $z ($details), redundancy: $red%"
    Log.print(session.logger, msg)
  end
  return report
end

function zeroth_iteration!(session::Session)
  @assert session.env.itc == 0
  Log.section(session.logger, 2, "Initial report")
  Report.print(session.logger, initial_report(session.env))
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
- `autosave`: if set to `false`, the session won't be saved automatically nor
    any file will be generated (default is `true`)
- `nostdout`: disables logging on the standard output when set to `true`
    (default is `false`)
- `benchmark`: vector of [`Benchmark.Duel`](@ref) to be used as a benchmark
    (default is `[]`)
- `load_saved_params`: if set to `true`, load the training parameters from
    the session directory (if present) rather than using the `params`
    argument (default is `false`)
"""
function Session(
    ::Type{Game}, ::Type{Net}, params, netparams;
    dir=nothing, autosave=true, nostdout=false, benchmark=[],
    load_saved_params=false
  ) where {Game, Net}
  logger = session_logger(dir, nostdout, autosave)
  if valid_session_dir(dir)
    env = load_env(Game, Net, logger, dir,
      params=(load_saved_params ? nothing : params))
    # The parameters must be unchanged
    same_json(x, y) = JSON3.write(x) == JSON3.write(y)
    same_json(env.params, params) || @info "Using modified parameters"
    @assert same_json(Network.hyperparams(env.bestnn), netparams)
    session = Session(env, dir, logger, autosave, benchmark)
    session.report = load_session_report(dir, env.itc)
  else
    network = Net(netparams)
    env = Env{Game}(params, network)
    session = Session(env, dir, logger, autosave, benchmark)
    Log.section(session.logger, 1, "Initializing a new AlphaZero environment")
    zeroth_iteration!(session)
  end
  return session
end

"""
    Session(::Type{Game}, ::Type{Network}, dir::String) where {Game, Net}

Load an existing session from a directory.

This constructor accepts the optional keyword arguments
`autosave`, `nostdout` and `benchmark`.
"""
function Session(
    ::Type{Game}, ::Type{Network}, dir::String;
    autosave=true, nostdout=false, benchmark=[]
  ) where {Game, Network}
  @assert valid_session_dir(dir)
  logger = session_logger(dir, nostdout, autosave)
  env = load_env(Game, Network, logger, dir)
  return Session(env, dir, logger, autosave, benchmark)
end

"""
    Session(env::Env[, dir])

Create a session from an initial environment.

- The iteration counter of the environment must be equal to 0
- If a session directory is provided, this directory must not exist yet

This constructor features the optional keyword arguments
`autosave`, `nostdout` and `benchmark`.
"""
function Session(
    env::Env, dir=nothing; autosave=true, nostdout=false, benchmark=[])
  @assert isnothing(dir) || !isdir(dir)
  @assert env.itc == 0
  logger = session_logger(dir, nostdout, autosave)
  session = Session(env, dir, logger, autosave, benchmark)
  zeroth_iteration!(session)
  return session
end

#####
##### Public interface
#####

function resume!(session::Session)
  try
    train!(session.env, session)
  catch e
    isa(e, InterruptException) || rethrow(e)
    Log.section(session.logger, 1, "Interrupted by the user")
  end
end

function save(session::Session, dir=session.dir)
  save_env(session.env, dir)
end

function explore(session::Session{<:Env{Game}}) where Game
  Log.section(session.logger, 1, "Starting interactive exploration")
  explorer = AlphaZero.Explorer(session.env, Game())
  AlphaZero.launch(explorer)
end

function play_game(session::Session{<:Env{Game}}) where Game
  net = Network.copy(session.env.bestnn, on_gpu=true, test_mode=true)
  mcts = MCTS.Env{Game}(net, nworkers=64)
  GI.interactive!(Game(), MCTS.AI(mcts, timeout=5.0), GI.Human())
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
  show_space_after_progress_bar(session)
  Report.print(session.logger, report)
  session.progress = nothing
end

function Handlers.memory_analyzed(session::Session, report)
  Log.section(session.logger, 2, "Memory Analysis")
  Report.print(session.logger, report)
end

function Handlers.learning_started(session::Session, initial_status)
  Log.section(session.logger, 2, "Starting learning")
  Log.section(session.logger, 3, "Optimizing the loss")
  Report.print(session.logger, initial_status, style=Log.BOLD)
end

function Handlers.learning_epoch(session::Session, report)
  Report.print(session.logger, report.status_after)
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
  show_space_after_progress_bar(session)
  avgz = fmt("+.2f", report.reward)
  wr = win_rate(report.reward)
  red = fmt(".1f", report.redundancy * 100)
  nnr = report.nn_replaced ? ", network replaced" : ""
  msg = "Average reward: $avgz (win rate of $wr%$nnr), redundancy: $red%"
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
##### Replay training
#####

function walk_iterations(::Type{G}, ::Type{N}, dir::String) where {G, N}
  n = 0
  while valid_session_dir(iterdir(dir, n))
    n += 1
  end
  return (load_env(G, N, Logger(devnull), iterdir(dir, i)) for i in 0:n-1)
end

function run_new_benchmark(
    session::Session{<:Env{G, N}}, name, benchmark
  ) where {G,N}
  old_env = session.env
  Log.section(session.logger, 1, "Computing new benchmark: $name")
  reports = Benchmark.Report[]
  @assert !isnothing(session.dir)
  for env in walk_iterations(G, N, session.dir)
    session.env = env
    push!(reports, run_benchmark(session))
  end
  session.env = old_env
  # Save and plot
  dir = joinpath(session.dir, name)
  isdir(dir) || mkpath(dir)
  open(joinpath(dir, BENCHMARK_FILE), "w") do io
    JSON2.pretty(io, JSON3.write(reports))
  end
  plot_benchmark(session.env.params, reports, dir)
  return
end
