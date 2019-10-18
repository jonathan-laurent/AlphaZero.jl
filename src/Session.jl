#####
##### Session management
#####

mutable struct Session{Env}
  env :: Env
  dir :: String
  logfile :: IO
  logger :: Logger
  autosave :: Bool
  validation :: Option{Validation}
  # Temporary state for logging
  progress :: Option{Progress}
  learning_checkpoint :: Option{Report.Checkpoint}

  function Session(env, dir, logfile, logger, autosave, validation=nothing)
    return new{typeof(env)}(
      env, dir, logfile, logger, autosave, validation, nothing, nothing)
  end
end

#####
##### Save and load sessions
#####

const NET_FILE         =  "net.data"
const MEM_FILE         =  "mem.data"
const ITC_FILE         =  "iter.txt"
const REPORT_FILE      =  "report.json"
const PARAMS_FILE      =  "params.json"
const NET_PARAMS_FILE  =  "netparams.json"
const VALIDATION_FILE  =  "validation.json"
const LOG_FILE         =  "log.txt"
const PLOTS_DIR        =  "plots"

iterdir(dir, i) = joinpath(dir, "$i")

function save_env(env::Env, dir)
  isdir(dir) || mkpath(dir)
  # Saving parameters
  open(joinpath(dir, PARAMS_FILE), "w") do io
    JSON2.pretty(io, JSON2.write(env.params))
  end
  open(joinpath(dir, NET_PARAMS_FILE), "w") do io
    JSON2.pretty(io, JSON2.write(hyperparams(env.bestnn)))
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

save(session::Session, dir=session.dir) = save_env(session.env, dir)

function valid_session_dir(dir)
  isfile(joinpath(dir, PARAMS_FILE)) &&
  isfile(joinpath(dir, NET_FILE)) &&
  isfile(joinpath(dir, MEM_FILE)) &&
  isfile(joinpath(dir, ITC_FILE))
end

function run_validation_experiment(session, idir)
  if !isnothing(session.validation)
    v = session.validation
    Log.section(session.logger, 2, "Running validation experiment")
    progress = Log.Progress(session.logger, length(v))
    report = validation_score(session.env, v, progress)
    show_space_after_progress_bar(session)
    z = fmt("+.2f", report.z)
    Log.print(session.logger, "Average reward: $z")
    if session.autosave
      isdir(idir) || mkpath(idir)
      open(joinpath(idir, VALIDATION_FILE), "w") do io
        JSON2.pretty(io, JSON2.write(report))
      end
    end
  end
end

function show_space_after_progress_bar(session)
  Log.console_only(session.logger) do
    Log.sep(session.logger, force=true)
  end
end

#####
##### Create and resume sessions
#####

# Start a new session
function Session(
    ::Type{Game}, ::Type{Network}, params, netparams;
    dir="session", autosave=true, autoload=true, validation=nothing
  ) where {Game, Network}
  autosave && (isdir(dir) || mkpath(dir))
  logfile = autosave ? open(joinpath(dir, LOG_FILE), "a") : devnull
  logger = Logger(logfile=logfile)
  if autoload && valid_session_dir(dir)
    env = load_env(Game, Network, logger, dir)
    session = Session(env, dir, logfile, logger, autosave, validation)
  else
    Log.section(logger, 1, "Initializing a new AlphaZero environment")
    network = Network(netparams)
    env = Env{Game}(params, network)
    session = Session(env, dir, logfile, logger, autosave, validation)
    run_validation_experiment(session, iterdir(session.dir, 0))
    if autosave
      save(session)
      save(session, iterdir(session.dir, 0))
    end
  end
  return session
end

# Load an existing session from a directory
function Session(
    ::Type{Game}, ::Type{Network}, dir; autosave=true, validation=nothing
  ) where {Game, Network}
  env = load_env(Game, Network, session.logger, dir)
  logfile = open(joinpath(dir, LOG_FILE), "a")
  logger = Logger(logfile=logfile)
  return Session(env, dir, logfile, logger, autosave, validation)
end

function resume!(session::Session)
  try
    train!(session.env, session)
  catch e
    isa(e, InterruptException) || rethrow(e)
    Log.section(session.logger, 1, "Interrupted by the user")
  end
end

function explore(session::Session{<:Env{Game}}) where Game
  Log.section(session.logger, 1, "Starting interactive exploration")
  explorer = AlphaZero.Explorer(session.env, Game())
  AlphaZero.launch(explorer)
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
  Report.print(session.logger, initial_status, style=Log.BOLD)
end

function Handlers.learning_checkpoint(session::Session, report)
  session.learning_checkpoint = report
end

function Handlers.learning_epoch(session::Session, report)
  comments = String[]
  checkpoint = session.learning_checkpoint
  if !isnothing(checkpoint)
    z = fmt("+.2f", checkpoint.reward)
    push!(comments, "Evaluation reward: $z")
    checkpoint.nn_replaced && push!(comments, "Networked replaced")
  end
  report.stable_loss && push!(comments, "Loss stabilized")
  Report.print(session.logger, report.status_after, comments)
  session.learning_checkpoint = nothing
end

function Handlers.learning_finished(session::Session, report)
  return
end

function Handlers.iteration_finished(session::Session, report)
  idir = iterdir(session.dir, session.env.itc)
  run_validation_experiment(session, idir)
  if session.autosave
    save(session)
    save(session, idir)
    open(joinpath(idir, REPORT_FILE), "w") do io
      JSON2.pretty(io, JSON2.write(report))
    end
    plot_report(session.dir)
    Log.section(session.logger, 2, "Environment saved in: $(session.dir)")
  end
  flush(session.logfile)
end

function Handlers.training_finished(session::Session)
  Log.section(session.logger, 1, "Training completed")
  close(session.logfile)
end

#####
##### Validation
#####

function walk_iterations(::Type{G}, ::Type{N}, dir::String) where {G, N}
  n = 0
  while valid_session_dir(iterdir(dir, n))
    n += 1
  end
  return (load_env(G, N, Logger(devnull), iterdir(dir, i)) for i in 0:n-1)
end

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

function get_reports(dir::String)
  n = 0
  ireps = Report.Iteration[]
  vreps = ValidationReport[]
  while valid_session_dir(iterdir(dir, n))
    idir = iterdir(dir, n)
    rep_file = joinpath(idir, REPORT_FILE)
    val_file = joinpath(idir, VALIDATION_FILE)
    if isfile(rep_file)
      open(rep_file, "r") do io
        push!(ireps, JSON2.read(io, Report.Iteration))
      end
    end
    if isfile(val_file)
      open(val_file, "r") do io
        push!(vreps, JSON2.read(io, ValidationReport))
      end
    end
    n += 1
  end
  length(ireps) == n - 1 || (ireps = nothing)
  length(vreps) == n || (vreps = nothing)
  # Load params file
  params_file = joinpath(dir, PARAMS_FILE)
  params = open(params_file, "r") do io
    JSON2.read(io, Params)
  end
  return params, ireps, vreps
end

function plot_report(dir::String)
  params, ireps, vreps = get_reports(dir)
  isnothing(ireps) && return
  plot_report(params, ireps, vreps, joinpath(dir, PLOTS_DIR))
end
