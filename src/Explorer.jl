#####
##### Computing board statistics
#####

@kwdef mutable struct BoardActionStats
  Pmcts :: Option{Float64} = nothing
  Qmcts :: Option{Float64} = nothing
  UCT   :: Option{Float64} = nothing
  Pmem  :: Option{Float64} = nothing
  Pnet  :: Float64 = 0.
  Qnet  :: Float64 = 0.
end

mutable struct BoardStats{Action}
  Nmcts :: Int
  Nmem  :: Int
  Vnet  :: Float64
  Vmem  :: Option{Float64}
  # Sorted by Nmcts
  actions :: Vector{Tuple{Action, BoardActionStats}}
  function BoardStats(actions)
    actions_stats = [(a, BoardActionStats()) for a in actions]
    new{eltype(actions)}(0, 0, 0., nothing, actions_stats)
  end
end

function evaluate_Qnet(oracle::MCTS.Oracle{G}, board, action) where G
  state = G(board)
  @assert isnothing(GI.white_reward(state))
  GI.play!(state, action)
  r = GI.white_reward(state)
  isnothing(r) || (return r)
  _, r = MCTS.evaluate(
    oracle, GI.canonical_board(state), GI.available_actions(state))
  GI.white_playing(state) || (r = MCTS.symmetric_reward(r))
  return r
end

function board_statistics(env::Env{G}, board) where G
  state = G(board)
  @assert isnothing(GI.white_reward(state))
  actions = GI.available_actions(state)
  report = BoardStats(actions)
  # Collect MCTS statistics
  if haskey(env.mcts.tree, board)
    info = env.mcts.tree[board]
    ucts = MCTS.uct_scores(info, env.mcts.cpuct)
    report.Nmcts = info.Ntot
    @assert actions == info.actions
    for (i, a) in enumerate(actions)
      astats = info.stats[i]
      arep = report.actions[i][2]
      arep.Pmcts = astats.N / max(1, info.Ntot)
      arep.Qmcts = astats.N > 0 ? astats.W / astats.N : 0.
      arep.UCT = ucts[i]
    end
  end
  # Collect memory statistics
  mem = merge_by_board(get(env.memory))
  relevant = findall((ex -> ex.b == board), mem)
  if !isempty(relevant)
    @assert length(relevant) == 1
    e = mem[relevant[1]]
    report.Nmem = e.n
    report.Vmem = e.z
    for i in eachindex(actions)
      report.actions[i][2].Pmem = e.π[i]
    end
  end
  # Evaluate the positions with the neural network
  Pnet, Vnet = MCTS.evaluate(env.bestnn, board, actions)
  report.Vnet = Vnet
  for i in eachindex(actions)
    arep = report.actions[i][2]
    arep.Pnet = Pnet[i]
    arep.Qnet = evaluate_Qnet(env.bestnn, board, actions[i])
  end
  # Sort the actions from best to worst
  if report.Nmcts > 0
    sortby = ((a, arep),) -> arep.Pmcts
  else
    sortby = ((a, arep),) -> arep.Pnet
  end
  sort!(report.actions, rev=true, by=sortby)
  return report
end

#####
##### Displaying board statistics
#####

function print_board_statistics(::Type{G}, stats::BoardStats) where G
  prob   = Log.ColType(nothing, x -> fmt(".2f", 100 * x) * "%")
  val    = Log.ColType(nothing, x -> fmt("+.2f", x))
  bigint = Log.ColType(nothing, n -> format(ceil(Int, n), commas=true))
  alabel = Log.ColType(nothing, identity)
  btable = Log.Table(
    ("Nmcts", bigint, r -> r.Nmcts),
    ("Nmem",  bigint, r -> r.Nmem),
    ("Vmem",  val,    r -> r.Vmem),
    ("Vnet",  val,    r -> r.Vnet),
    header_style=Log.BOLD)
  atable = Log.Table(
    ("",      alabel, r -> GI.action_string(G, r[1])),
    ("Pmcts", prob,   r -> r[2].Pmcts),
    ("Pnet",  prob,   r -> r[2].Pnet),
    ("UCT",   val,    r -> r[2].UCT),
    ("Pmem",  prob,   r -> r[2].Pmem),
    ("Qmcts", val,    r -> r[2].Qmcts),
    ("Qnet",  val,    r -> r[2].Qnet),
    header_style=Log.BOLD)
  logger = Logger()
  #Log.print(logger, "Board statistics")
  #Log.sep(logger)
  Log.table(logger, btable, [stats])
  Log.sep(logger)
  #Log.print(logger, "Action statistics")
  #Log.sep(logger)
  Log.table(logger, atable, stats.actions)
  Log.sep(logger)
end

function print_state_statistics(env::Env{G}, state::G) where G
  board = GI.canonical_board(state)
  print_board_statistics(G, board_statistics(env, board))
end

#####
##### Interactive exploration of the environment
#####

mutable struct Explorer{Game}
  env :: Env{Game}
  state :: Game
  history
  function Explorer(env, state)
    G = typeof(state)
    new{G}(env, state, Stack{Any}())
  end
end

save_state!(exp::Explorer) = push!(exp.history, copy(exp.state))

function interpret!(exp::Explorer{G}, cmd, args=[]) where G
  if cmd == "go"
    st = GI.read_state(G)
    if !isnothing(st)
      save_state!(exp)
      exp.state = st
      return true
    end
  elseif cmd == "undo"
    if !isempty(exp.history)
      exp.state = pop!(exp.history)
      return true
    end
  elseif cmd == "do"
    length(args) == 1 || return false
    a = GI.parse_action(exp.state, args[1])
    isnothing(a) && return false
    a ∈ GI.available_actions(exp.state) || return false
    save_state!(exp)
    GI.play!(exp.state, a)
    return true
  elseif cmd == "explore"
    try
      if isempty(args)
        n = exp.env.params.self_play.num_mcts_iters_per_turn
      else
        n = parse(Int, args[1])
      end
      MCTS.explore!(exp.env.mcts, exp.state, n)
      return true
    catch e
      isa(e, ArgumentError) && (return false)
      rethrow(e)
    end
  end
  return false
end

function launch(exp::Explorer)
  while true
    # Print the state
    GI.print_state(exp.state)
    print_state_statistics(exp.env, exp.state)
    # Interpret command
    while true
      print("> ")
      inp = readline() |> lowercase |> split
      isempty(inp) && return
      cmd = inp[1]
      args = inp[2:end]
      interpret!(exp, cmd, args) && break
    end
    println("")
  end
end
