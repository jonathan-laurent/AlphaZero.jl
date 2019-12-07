#####
##### Session management
#####

"""
    SessionReport

The full collection of benchmarks and statistics
collected during a training session.
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

A basic user interface for AlphaZero environments.
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
    JSON2.pretty(io, JSON2.write(env.params))
  end
  open(joinpath(dir, NET_PARAMS_FILE), "w") do io
    JSON2.pretty(io, JSON2.write(Network.hyperparams(env.bestnn)))
  end
  # Saving state
  serialize(joinpath(dir, NET_FILE), env.bestnn)
  serialize(joinpath(dir, MEM_FILE), get(env.memory))
  open(joinpath(dir, ITC_FILE), "w") do io
    JSON2.write(io, env.itc)
  end
end

function load_env(
    ::Type{Game}, ::Type{Network}, logger, dir) where {Game, Network}
  Log.section(logger, 1, "Loading environment")
  # Load parameters
  params_file = joinpath(dir, PARAMS_FILE)
  params = open(params_file, "r") do io
    JSON2.read(io, Params)
  end
  Log.print(logger, "Loading parameters from: $(params_file)")
  # Try to load network or otherwise network params
  net_file = joinpath(dir, NET_FILE)
  netparams_file = joinpath(dir, NET_PARAMS_FILE)
  if isfile(net_file)
    network = deserialize(net_file)
    Log.print(logger, "Loading network from: $(net_file)")
  else
    network = open(netparams_file, "r") do io
      params = JSON2.read(io, HyperParams(Network))
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
      JSON2.read(io)
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
      push!(rep.benchmark, JSON2.read(io, Benchmark.Report))
    end
    # Load the iteration report
    if itc > 0
      isfile(ifile) || error("Not found: $ifile")
      open(ifile, "r") do io
        push!(rep.iterations, JSON2.read(io, Report.Iteration))
      end
    end
  end
  @assert valid_session_report(rep)
  return rep
end

function save_report_increment(session, bench, itrep, idir)
  open(joinpath(idir, BENCHMARK_FILE), "w") do io
    JSON2.pretty(io, JSON2.write(bench))
  end
  if session.env.itc > 0
    @assert !isnothing(itrep)
    open(joinpath(idir, REPORT_FILE), "w") do io
      JSON2.pretty(io, JSON2.write(itrep))
    end
  end
end

#####
##### Incremental saving
#####

autosave_enabled(s::Session) = s.autosave && !isnothing(s.dir)

function save_increment(session::Session, bench, itrep=nothing)
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
    plot_training(params,
      session.report.iterations, session.report.benchmark,
      joinpath(session.dir, PLOTS_DIR))
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
    Log.print(session.logger, "Average reward: $z ($details)")
  end
  return report
end

function zeroth_iteration(session::Session)
  @assert session.env.itc == 0
  Log.section(session.logger, 1, "Initializing a new AlphaZero environment")
  Log.section(session.logger, 2, "Initial report")
  Report.print(session.logger, initial_report(session.env))
  bench = run_benchmark(session)
  save_increment(session, bench)
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
    Session(env::Env)

Initialize a session from an environment
"""
function Session(
    env::Env, dir=nothing; autosave=true, nostdout=false, benchmark=[])
  logger = session_logger(dir, nostdout, autosave)
  return Session(env, dir, logger, autosave, benchmark)
end

"""
    Session(::Type{Game}, ::Type{Network}, params, netparams)

Create a new session.
"""
function Session(
    ::Type{Game}, ::Type{Net}, params, netparams;
    dir=nothing, autosave=true, nostdout=false, benchmark=[]
  ) where {Game, Net}
  logger = session_logger(dir, nostdout, autosave)
  if valid_session_dir(dir)
    env = load_env(Game, Net, logger, dir)
    # The parameters must be unchanged
    same_json(x, y) = JSON2.write(x) == JSON2.write(y)
    @assert same_json(env.params, params)
    @assert same_json(Network.hyperparams(env.bestnn), netparams)
    session = Session(env, dir, logger, autosave, benchmark)
    session.report = load_session_report(dir, env.itc)
  else
    network = Net(netparams)
    env = Env{Game}(params, network)
    session = Session(env, dir, logger, autosave, benchmark)
    zeroth_iteration(session)
  end
  return session
end

"""
    Session(::Type{Game}, ::Type{Network}, dir::String)

Load an existing session from a directory.
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
##### Run benchmarks
#####


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
  nnr = report.nn_replaced ? ", network replaced" : ""
  Log.print(session.logger, "Average reward: $avgz (win rate of $wr%$nnr)")
  Log.section(session.logger, 3, "Optimizing the loss")
end

function Handlers.learning_finished(session::Session, report)
  return
end

function Handlers.iteration_finished(session::Session, report)
  bench = run_benchmark(session)
  save_increment(session, bench, report)
  flush(Log.logfile(session.logger))
end

function Handlers.training_finished(session::Session)
  Log.section(session.logger, 1, "Training completed")
  close(Log.logfile(session.logger))
end

#####
##### Time travel
#####

#=
function walk_iterations(::Type{G}, ::Type{N}, dir::String) where {G, N}
  n = 0
  while valid_session_dir(iterdir(dir, n))
    n += 1
  end
  return (load_env(G, N, Logger(devnull), iterdir(dir, i)) for i in 0:n-1)
end
=#
#=
function validate(::Type{G}, ::Type{N}, dir::String, v) where {G, N}
  logger = Logger()
  Log.section(logger, 1, "Running validation experiment")
  for env in walk_iterations(G, N, dir)
    Log.section(logger, 2, "Iteration $(env.itc)")
    progress = Log.Progress(logger, v.num_games)
    report = validation_score(env, v, progress)
    show_space_after_progress_bar(session)
    z = fmt("+.2f", report.z)
    Log.print(logger, "Average reward: $z")
  end
end
=#
