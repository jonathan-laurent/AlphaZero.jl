#####
##### Exploring a player's policy
#####

using DataStructures: Stack
import AlphaZero: MemoryBuffer

# All state statistics and are expressed with respect to the current player

@kwdef mutable struct StateActionStats
  P     :: Option{Float64} = nothing  # Probability given by `think`
  Pmem  :: Option{Float64} = nothing  # Recorded π component in memory
  Pmcts :: Option{Float64} = nothing  # Percentage of MCTS visits
  Qmcts :: Option{Float64} = nothing
  UCT   :: Option{Float64} = nothing
  Pnet  :: Option{Float64} = nothing
  Qnet  :: Option{Float64} = nothing
end

mutable struct StateStats{Action}
  Nmcts :: Option{Int}
  Nmem  :: Option{Int}
  Vmem  :: Option{Float64}
  Vnet  :: Option{Float64}
  actions :: Vector{Tuple{Action, StateActionStats}} # Sorted by P
  function StateStats(actions)
    actions_stats = [(a, StateActionStats()) for a in actions]
    new{eltype(actions)}(nothing, nothing, nothing, nothing, actions_stats)
  end
end

player_reward(game, white_reward) =
  GI.white_playing(game) ? white_reward : - white_reward

function evaluate_vnet(oracle::MCTS.Oracle, game)
  if GI.game_terminated(game)
    return 0.0
  else
    return oracle(GI.current_state(game))[2]
  end
end

function evaluate_qnet(oracle::MCTS.Oracle, game, action, gamma)
  @assert !GI.game_terminated(game)
  next = copy(game)
  GI.play!(next, action)
  wr = GI.white_reward(next)
  r = GI.white_playing(game) ? wr : -wr
  qnext = evaluate_vnet(oracle, next)
  if GI.white_playing(game) != GI.white_playing(next)
    qnext = -qnext
  end
  return r + gamma * qnext
end

player_oracle(p) = nothing
player_oracle(p::MctsPlayer) = p.mcts.oracle
player_oracle(p::NetworkPlayer) = p.network

player_gamma(p) = p.gamma
player_gamma(p::MctsPlayer) = p.mcts.gamma

function state_statistics(game, player, turn, memory=nothing)
  @assert !GI.game_terminated(game)
  state = GI.current_state(game)
  # Make the player think
  actions, π = think(player, game)
  τ = player_temperature(player, game, turn)
  π = apply_temperature(π, τ)
  report = StateStats(actions)
  for i in eachindex(actions)
    report.actions[i][2].P = π[i]
  end
  # Collect MCTS Statistics
  if isa(player, MctsPlayer) && haskey(player.mcts.tree, state)
    mcts = player.mcts
    info = mcts.tree[state]
    ucts = MCTS.uct_scores(info, mcts.cpuct, 0., nothing)
    report.Nmcts = MCTS.Ntot(info)
    for (i, a) in enumerate(actions)
      astats = info.stats[i]
      arep = report.actions[i][2]
      arep.Pmcts = astats.N / max(1, report.Nmcts)
      arep.Qmcts = astats.N > 0 ? astats.W / astats.N : 0.
      arep.UCT = ucts[i]
    end
  end
  # Collect memory statistics
  if !isnothing(memory)
    mem = AlphaZero.merge_by_state(get_experience(memory))
    relevant = findall((ex -> ex.s == state), mem)
    if !isempty(relevant)
      @assert length(relevant) == 1
      e = mem[relevant[1]]
      report.Nmem = e.n
      report.Vmem = e.z
      for i in eachindex(actions)
        report.actions[i][2].Pmem = e.π[i]
      end
    end
  end
  # Collect network statistics
  oracle = player_oracle(player)
  if isa(oracle, AbstractNetwork)
    Pnet, Vnet = oracle(state)
    report.Vnet = Vnet
    for i in eachindex(actions)
      arep = report.actions[i][2]
      arep.Pnet = Pnet[i]
      arep.Qnet = evaluate_qnet(oracle, game, actions[i], player_gamma(player))
    end
  end
  # Sort the actions from best to worst
  sortby = ((a, arep),) -> arep.P
  sort!(report.actions, rev=true, by=sortby)
  return report
end

#####
##### Displaying state statistics
#####

function print_state_statistics(::Type{G}, stats::StateStats) where G
  prob   = Log.ColType(nothing, x -> fmt(".1f", 100 * x) * "%")
  val    = Log.ColType(nothing, x -> fmt("+.2f", x))
  bigint = Log.ColType(nothing, n -> format(ceil(Int, n), commas=true))
  alabel = Log.ColType(nothing, identity)
  btable = Log.Table([
    ("Nmcts", bigint, r -> r.Nmcts),
    ("Nmem",  bigint, r -> r.Nmem),
    ("Vmem",  val,    r -> r.Vmem),
    ("Vnet",  val,    r -> r.Vnet)],
    header_style=Log.BOLD)
  atable = Log.Table([
    ("",      alabel, r -> GI.action_string(G, r[1])),
    ("",      prob,   r -> r[2].P),
    ("Pmcts", prob,   r -> r[2].Pmcts),
    ("Pnet",  prob,   r -> r[2].Pnet),
    ("UCT",   val,    r -> r[2].UCT),
    ("Pmem",  prob,   r -> r[2].Pmem),
    ("Qmcts", val,    r -> r[2].Qmcts),
    ("Qnet",  val,    r -> r[2].Qnet)],
    header_style=Log.BOLD)
  logger = Logger()
  if !all(isnothing, [stats.Nmcts, stats.Nmem, stats.Vmem, stats.Vnet])
    Log.table(logger, btable, [stats])
    Log.sep(logger)
  end
  Log.table(logger, atable, stats.actions)
  Log.sep(logger)
end

# Return the stats
function compute_and_print_state_statistics(exp)
  if !GI.game_terminated(exp.game)
    stats = state_statistics(exp.game, exp.player, exp.turn, exp.memory)
    print_state_statistics(GameType(exp), stats)
    return stats
  else
    return nothing
  end
end

#####
##### Interactive exploration of the environment
#####

"""
    Explorer{Game}

A command interpreter to explore the internals of a player
through interactive play.

# Constructors

    Explorer(player::AbstractPlayer, game=nothing; memory=nothing)

Build an explorer to investigate the behavior of `player` in a given `game`
(by default, in the initial state). Optionally, a reference to a memory buffer
can be provided, in which case additional state statistics
will be displayed.

    Explorer(env::Env, game=nothing; arena_mode=false)

Build an explorer for the MCTS player based on neural network `env.bestnn`
and on parameters `env.params.self_play.mcts` or `env.params.arena.mcts`
(depending on the value of `arena_mode`).

# Commands

The following commands are currently implemented:

  - `do [action]`: make the current player perform `action`.
    By default, the action of highest score is played.
  - `explore [num_sims]`: run `num_sims` MCTS simulations from the current
    state (for MCTS players only).
  - `go`: query the user for a state description and go to this state.
  - `flip`: flip the board according to a random symmetry.
  - `undo`: undo the effect of the previous command.
  - `restart`: restart the explorer.
"""
mutable struct Explorer{Game}
  game :: Game
  history :: Stack{Game}
  player :: AbstractPlayer{Game}
  memory :: Option{MemoryBuffer}
  turn :: Int
  function Explorer(player::AbstractPlayer, game=nothing; memory=nothing)
    Game = GameType(player)
    isnothing(game) && (game = Game())
    history = Stack{Game}()
    new{Game}(game, history, player, memory, 0)
  end
end

GameType(::Explorer{Game}) where Game = Game

function Explorer(
    env::Env, game=nothing; mcts_params=env.params.self_play.mcts, on_gpu=false)
  Game = GameType(env)
  isnothing(game) && (game = Game())
  net = Network.copy(env.bestnn, on_gpu=on_gpu, test_mode=true)
  player = MctsPlayer(net, mcts_params)
  return Explorer(player, game, memory=env.memory)
end

function restart!(exp::Explorer)
  reset_player!(exp.player)
  empty!(exp.history)
  exp.game = GameType(exp)()
  exp.turn = 0
end

save_game!(exp::Explorer) = push!(exp.history, copy(exp.game))

# Return true if the command is valid, false otherwise
function interpret!(exp::Explorer, stats, cmd, args=[])
  Game = GameType(exp)
  if cmd == "go"
    st = GI.read_state(Game)
    if isnothing(st)
      println("Invalid state description.")
      return false
    end
    g = Game(st)
    if !isnothing(st)
      save_game!(exp)
      exp.game = g
      exp.turn = 0
      return true
    end
  elseif cmd == "restart"
    restart!(exp)
    return true
  elseif cmd == "undo"
    if !isempty(exp.history)
      exp.game = pop!(exp.history)
      exp.turn -= 1
      return true
    end
  elseif cmd == "do"
    GI.game_terminated(exp.game) && return false
    if length(args) == 0
      a = stats.actions[1][1]
    else
      length(args) == 1 || return false
      a = GI.parse_action(exp.game, args[1])
      isnothing(a) && return false
      a ∈ GI.available_actions(exp.game) || return false
    end
    save_game!(exp)
    GI.play!(exp.game, a)
    exp.turn += 1
    return true
  elseif cmd == "flip"
    save_game!(exp)
    exp.game = GI.apply_random_symmetry(exp.game)
    return true
  elseif cmd == "explore"
    isa(exp.player, MctsPlayer) || (return false)
    try
      if isempty(args)
        n = exp.player.niters
      else
        n = parse(Int, args[1])
      end
      MCTS.explore!(exp.player.mcts, exp.game, n)
      return true
    catch e
      isa(e, ArgumentError) && (return false)
      rethrow(e)
    end
  end
  return false
end

"""
    start_explorer(exp::Explorer)

Start an interactive explorer session.
"""
function start_explorer(exp::Explorer)
  while true
    # Print the state
    GI.render(exp.game)
    stats = compute_and_print_state_statistics(exp)
    # Interpret command
    while true
      print("> ")
      inp = readline() |> lowercase |> split
      isempty(inp) && return
      cmd = inp[1]
      args = inp[2:end]
      interpret!(exp, stats, cmd, args) && break
    end
    println("")
  end
end
