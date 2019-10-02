#####
##### Inspecting the experience buffer
#####

function inspect_memory(env::Env{G}, state::G) where G
  mem = get(env.memory)
  board = GI.canonical_board(state)
  relevant = findall((ex -> ex.b == board), mem)
  if isempty(relevant)
    println("Not in memory\n")
  else
    @assert length(relevant) == 1
    e = mem[relevant[1]]
    @printf("N: %d, z: %.4f\n\n", e.n, e.z)
    as = GI.available_actions(state)
    for (i, p) in sort(collect(enumerate(e.π)), by=(((i,p),)->p), rev=true)
      @printf("%1s %6.3f\n", GI.action_string(G, as[i]), p)
    end
  end
  println("")
end

function viz_memory(env)
  mem = get(env.memory)
  ns = [e.n for e in mem]
  println("Number of distinct board configurations: $(length(ns))")
  Plots.histogram(ns, weights=ns, legend=nothing)
end


#####
##### Interactive exploration of the environment
#####

mutable struct Explorer{Game}
  env :: Env{Game}
  state :: Game
  history
  mcts
  oracle
  function Explorer(env, state, mcts=env.mcts, oracle=env.bestnn)
    new{typeof(state)}(env, state, Stack{Any}(), mcts, oracle)
  end
end

save_state!(exp::Explorer) = push!(exp.history, copy(exp.state))

function print_state_statistics(exp::Explorer{G}) where G
  board = GI.canonical_board(exp.state)
  if haskey(exp.mcts.tree, board)
    info = exp.mcts.tree[board]
    if !isnothing(exp.oracle)
      Pnet, Vnet = MCTS.evaluate(exp.oracle, board, info.actions)
    end
    @printf("N: %d, V: %.3f", info.Ntot, info.Vest)
    isnothing(exp.oracle) || @printf(", Vnet: %.3f", Vnet)
    @printf("\n\n")
    actions = enumerate(info.actions) |> collect
    actions = sort(actions, by=(((i,a),) -> info.stats[i].N), rev=true)
    ucts = MCTS.uct_scores(info, exp.mcts.cpuct)
    @printf("%1s %7s %8s %6s %8s ", "", "N (%)", "Q", "P", "UCT")
    isnothing(exp.oracle) || @printf("%8s", "Pnet")
    @printf("\n")
    for (i, a) in actions
      stats = info.stats[i]
      Nr = 100 * stats.N / info.Ntot
      Q = stats.N > 0 ? stats.W / stats.N : 0.
      astr = GI.action_string(G, a)
      @printf("%1s %7.2f %8.4f %6.2f %8.4f ", astr, Nr, Q, stats.P, ucts[i])
      isnothing(exp.oracle) || @printf("%8.4f", Pnet[i])
      @printf("\n")
    end
  else
    print("Unexplored board.")
  end
  println("")
end

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
    GI.print_state(exp.state)
    print_state_statistics(exp)
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
