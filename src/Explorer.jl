################################################################################
# Explorer.jl
################################################################################

function inspect_memory(env, board::Board)
  mem = get(env.memory)
  relevant = findall((ex -> ex.b == board), mem)
  isempty(relevant) && (return nothing)
  @assert length(relevant) == 1
  return mem[relevant[1]]
end

function inspect_memory(env, state::State)
  b = GI.canonical_board(state)
  e = inspect_memory(env, b)
  if isnothing(e)
    println("Not in memory\n")
  else
    @printf("N: %d, z: %.4f\n\n", e.n, e.z)
    as = GI.available_actions(state)
    for (i, p) in sort(collect(enumerate(e.π)), by=(((i,p),)->p), rev=true)
      @printf("%1s %6.3f\n", action_str(as[i]), p)
    end
  end
  println("")
end

import Plots

function show_memory_stats(env)
  mem = get(env.memory)
  ns = [e.n for e in mem]
  println("Number of distinct board configurations: $(length(ns))")
  Plots.histogram(ns, weights=ns, legend=nothing)
end

################################################################################

function input_board()
  str = reduce(*, ((readline() * "   ")[1:3] for i in 1:3))
  white = ['w', 'r', 'o']
  black = ['b', 'b', 'x']
  board = TicTacToe.make_board()
  for i in 1:9
    c = nothing
    str[i] ∈ white && (c = Red)
    str[i] ∈ black && (c = Blue)
    board[i,1] = c
  end
  return board
end

# Enter a state from command line (returns `nothing` if invalid)
function input_state()
  b = input_board()
  nr = count(==(Red), b[:,1])
  nb = count(==(Blue), b[:,1])
  if nr == nb # red turn
    State(b, first_player=Red)
  elseif nr == nb + 1
    State(b, first_player=Blue)
  else
    nothing
  end
end

function action_str(a)
  TicTacToe.print_pos(a.to)
end

################################################################################

function print_state_statistics(mcts, state, oracle = nothing)
  wp = GI.white_playing(state)
  b = GI.canonical_board(state)
  if haskey(mcts.tree, b)
    info = mcts.tree[b]
    if !isnothing(oracle)
      board = GI.canonical_board(state)
      Pnet, Vnet = MCTS.evaluate(oracle, board, info.actions)
    end
    @printf("N: %d, V: %.3f", info.Ntot, info.Vest)
    isnothing(oracle) || @printf(", Vnet: %.3f", Vnet)
    @printf("\n\n")
    actions = enumerate(info.actions) |> collect
    actions = sort(actions, by=(((i,a),) -> info.stats[i].N), rev=true)
    ucts = MCTS.uct_scores(info, mcts.cpuct)
    @printf("%1s %7s %8s %6s %8s ", "", "N (%)", "Q", "P", "UCT")
    isnothing(oracle) || @printf("%8s", "Pnet")
    @printf("\n")
    for (i, a) in actions
      stats = info.stats[i]
      Nr = 100 * stats.N / info.Ntot
      Q = stats.N > 0 ? stats.W / stats.N : 0.
      astr = action_str(a)
      @printf("%1s %7.2f %8.4f %6.2f %8.4f ", astr, Nr, Q, stats.P, ucts[i])
      isnothing(oracle) || @printf("%8.4f", Pnet[i])
      @printf("\n")
    end
  else
    print("Unexplored board.")
  end
  println("")
end

################################################################################

using DataStructures: Stack

mutable struct Explorer
  env
  state
  history
  mcts
  oracle
  Explorer(env, state, mcts=env.mcts, oracle=env.bestnn) =
    new(env, state, Stack{Any}(), mcts, oracle)
end

save_state!(exp::Explorer) = push!(exp.history, deepcopy(exp.state))

################################################################################

function interpret!(exp::Explorer, cmd, args=[])
  if cmd == "go"
    st = input_state()
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
    a = TicTacToe.parse_action(exp.state, args[1])
    isnothing(a) && return false
    a ∈ GI.available_actions(exp.state) || return false
    save_state!(exp)
    GI.play!(exp.state, a)
    return true
  elseif cmd == "mem"
    inspect_memory(exp.env, exp.state)
    return false
  elseif cmd == "explore"
    try
      if isempty(args)
        n = exp.env.params.self_play.num_mcts_iters_per_turn
      else
        n = parse(Int, args[1])
      end
      MCTS.explore!(exp.mcts, exp.state, n)
      return true
    catch ArgumentError return false end
  end
  return false
end

function launch(exp::Explorer)
  while true
    # Print the state
    println("")
    TicTacToe.print_board(exp.state, with_position_names=true)
    println("")
    print_state_statistics(exp.mcts, exp.state, exp.oracle)
    # Interpret command
    while true
      print("> ")
      inp = readline() |> lowercase |> split
      isempty(inp) && return
      cmd = inp[1]
      args = inp[2:end]
      interpret!(exp, cmd, args) && break
    end
  end
end

################################################################################
