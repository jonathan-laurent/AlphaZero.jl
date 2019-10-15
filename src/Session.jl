#####
##### Session management
#####

mutable struct Session{Env}
  env :: Env
  dir :: String
  logger :: Logger
  autosave :: Bool
  # Temporary state for logging
  progress :: Option{Progress}
  learning_checkpoint :: Option{Report.Checkpoint}
  
  function Session(env, dir, logger, autosave)
    return new{typeof(env)}(
      env, dir, logger, autosave, nothing, nothing)
  end
end

#####
##### Save and load sessions
#####

const NET_FILE         =  "net.data"
const MEM_FILE         =  "mem.data"
const ITC_FILE         =  "iter.txt"
const REP_FILE         =  "report.json"
const PARAMS_FILE      =  "params.json"
const NET_PARAMS_FILE  =  "netparams.json"

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

#####
##### Create and resume sessions
#####

# Start a new session
function Session(
    ::Type{Game}, ::Type{Network}, params, netparams;
    dir="session", autosave=true, autoload=true
  ) where {Game, Network}
  logger = Logger()
  if autoload && valid_session_dir(dir)
    env = load_env(Game, Network, logger, dir)
    session = Session(env, dir, logger, autosave)
  else
    Log.section(logger, 1, "Initializing a new AlphaZero environment")
    network = Network(netparams)
    env = Env{Game}(params, network)
    session = Session(env, dir, logger, autosave)
    if autosave
      save(session)
      save(session, iterdir(session.dir, 0))
    end
  end
  return session
end

# Load an existing session from a directory
function Session(
  ::Type{Game}, ::Type{Network}, dir; autosave=true) where {Game, Network}
  env = load_env(Game, Network, session.logger, dir)
  logger = Logger()
  return Session(env, dir, logger, autosave)
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
  indent = repeat(" ", Log.offset(session.logger))
  session.progress = Progress(ngames, desc=(indent * "Progress: "))
end

function Handlers.game_played(session::Session)
  next!(session.progress)
end

function Handlers.self_play_finished(session::Session, report)
  Log.print(session.logger, "")
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
  if session.autosave
    save(session)
    idir = iterdir(session.dir, session.env.itc)
    save(session, idir)
    open(joinpath(idir, REP_FILE), "w") do io
      JSON2.pretty(io, JSON2.write(report))
    end
    Log.section(session.logger, 2, "Environment saved in: $(session.dir)")
  end
end

function Handlers.training_finished(session::Session)
  Log.section(session.logger, 1, "Training completed")
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
    indent = repeat(" ", Log.offset(logger))
    ngames = v.num_games
    progress = Progress(ngames, desc=(indent * "Progress: "))
    report = validation_score(env, v, progress)
    Log.print(logger, "")
    z = fmt("+.2f", report.z)
    Log.print(logger, "Average reward: $z")
  end
end
