################################################################################
# Setting up the network
################################################################################

using Printf
using AlphaZero.GameInterface
using AlphaZero.MCTS
using Distributions: Categorical

################################################################################
# Params

using Base: @kwdef
import AlphaZero.Util.Records

struct Params
  num_learning_iters       :: Int
  num_episodes_per_iter    :: Int
  num_mcts_iters_per_turn  :: Int
  mem_buffer_size          :: Int
  num_eval_games           :: Int
  update_threshold         :: Float64
  dirichlet_noise_n        :: Float64  # Dir(α) with α=n/num_actions
  dirichlet_noise_ϵ        :: Float64
  cpuct                    :: Float64
end

Records.generate_named_constructors(Params) |> eval

# Some standard values for params:
# https://github.com/suragnair/alpha-zero-general/blob/master/main.py
# For dirichlet noise, see:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3

const STD_PARAMS = Params(
  num_learning_iters       = 1_000,
  num_episodes_per_iter    = 1_000,  # 100
  num_mcts_iters_per_turn  = 100,  # 25 in Nair's implementation
  mem_buffer_size          = 200_000,
  num_eval_games           = 40,
  update_threshold         = 0.6,
  dirichlet_noise_n        = 10,
  dirichlet_noise_ϵ        = 0.25,
  cpuct                    = 1.0)

# Alpha go test
# 4.9 million games of self play
# Parameters updated from 700,000 minibatches of 2048 positions
# Neural network: 20 residual blocks
# momenum param: 0.9
# Checkpoint after 1000 training steps
# First 30 moves, τ=1, then τ → 0
# Question: when pitting network against each other,
# where does randomness come from?
const ALPHA_GO_ZERO_PARAMS = Params(
  num_learning_iters       = 200, # = 5M/25K
  num_episodes_per_iter    = 25_000,
  num_mcts_iters_per_turn  = 1600, # 0.4s thinking time per move
  mem_buffer_size          = 500_000,
  num_eval_games           = 400,
  update_threshold         = 0.55,
  dirichlet_noise_n        = 10,
  dirichlet_noise_ϵ        = 0.25,
  cpuct                    = 1.0)

# Alpha go final:
# 29 million games, 31 million minibatches of 2048

const TAU_EPS = 0.1

################################################################################

using Flux

const R = Float32

struct Network
  common
  vbranch
  pbranch
  function Network(indim, outdim, hsize)
    common = Chain(
      Dense(indim, hsize, relu),
      Dense(hsize, hsize, relu),
      Dense(hsize, hsize, relu),
      Dense(hsize, hsize, relu))
    vbranch = Chain(
      Dense(hsize, hsize, relu),
      Dense(hsize, 1, tanh))
    pbranch = Chain(
      Dense(hsize, hsize, relu),
      Dense(hsize, outdim),
      softmax)
    new(common, vbranch, pbranch)
  end
end

Flux.@treelike Network

# Define the forward pass
function (nn::Network)(board, actions_mask)
  c = nn.common(board)
  v = nn.vbranch(c)
  p = nn.pbranch(c) .* actions_mask
  sp = sum(p)
  @assert sp > 0
  p = p ./ sp
  return (p, v)
end

################################################################################
# Interface to the neural network

struct Oracle{Game} <: MCTS.Oracle
  nn :: Network
  function Oracle{G}() where G
    hsize = 100
    nn = Network(GI.board_dim(G), GI.num_actions(G), hsize)
    new(nn)
  end
end

num_parameters(oracle) = sum(length(p) for p in params(o.nn))

function Base.copy(o::Oracle{G}) where G
  new = Oracle{G}()
  Flux.loadparams!(new.nn, params(o.nn))
  return new
end

function MCTS.evaluate(o::Oracle{G}, board, available_actions) where G
  mask = GI.actions_mask(G, available_actions)
  input = GI.vectorize_board(G, board)
  P, V = o.nn(input, mask)
  P = P[mask]
  return Vector{Float64}(Tracker.data(P)), Float64(Tracker.data(V)[1])
end

################################################################################
# Memory Buffer
# Structure for collecting experience

using DataStructures: CircularBuffer

struct TrainingExample{Board}
  b :: Board
  π :: Vector{Float64}
  z :: Float64
  w :: Float64 # Sample weights
end

struct MemoryBuffer{Board}
  # State-policy pairs accumulated during the current game.
  # The z component is 1 if white was playing and -1 otherwise
  cur :: Vector{TrainingExample{Board}}
  # Long-term memory
  mem :: CircularBuffer{TrainingExample{Board}}

  function MemoryBuffer{B}(size) where B
    new{B}([], CircularBuffer{TrainingExample{B}}(size))
  end
end

get(b::MemoryBuffer) = b.mem[:]

Base.length(b::MemoryBuffer) = size(b.mem)

function push_sample!(buf::MemoryBuffer, board, policy, white_playing)
  player_code = white_playing ? 1.0 : -1.0
  ex = TrainingExample(board, policy, player_code, 1.0)
  push!(buf.cur, ex)
end

function push_game!(buf::MemoryBuffer, white_reward)
  for ex in buf.cur
    r = ex.z * white_reward
    push!(buf.mem, TrainingExample(ex.b, ex.π, r, ex.w))
  end
  empty!(buf.cur)
end

################################################################################

weighted_mse(ŷ, y, w) = sum((ŷ .- y).^2 .* w) * 1 // length(y)

function random_minibatch(W, X, A, P, V; batchsize)
  n = size(X, 2)
  indices = rand(1:n, batchsize)
  return (W[:,indices], X[:,indices], A[:,indices], P[:,indices], V[:,indices])
end

function convert_sample(Game, e::TrainingExample)
  w = [R(e.w)]
  x = Vector{R}(GI.vectorize_board(Game, e.b))
  actions = GI.available_actions(Game(e.b))
  a = GI.actions_mask(Game, actions)
  p = zeros(R, size(a))
  p[[GI.action_id(Game, a) for a in actions]] = e.π
  v = [R(e.z)]
  return (w, x, a, p, v)
end

function convert_samples(Game, es::Vector{<:TrainingExample})
  ces = [convert_sample(Game, e) for e in es]
  W = reduce(hcat, (e[1] for e in ces))
  X = reduce(hcat, (e[2] for e in ces))
  A = reduce(hcat, (e[3] for e in ces))
  P = reduce(hcat, (e[4] for e in ces))
  V = reduce(hcat, (e[5] for e in ces))
  return (W, X, A, P, V)
end

function train!(
    oracle::Oracle{G},
    examples::Vector{<:TrainingExample};
    batch_size = 32,
    num_batches = 10_000,
    loss_eps = 1e-3,
    lr = 1e-3
  ) where G
  
  W, X, A, P, V = convert_samples(G, examples)
  function loss(W, X, A, P₀, V₀)
    let (P, V) = oracle.nn(X, A)
      Lp = Flux.crossentropy(P .+ eps(R), P₀, weight = W)
      Lv = weighted_mse(V, V₀, W)
      return Lp + Lv
    end
  end
  function print_legend()
    @printf("%12s", "Loss\n")
  end
  prevloss = Inf32
  function cb()
    L = loss(W, X, A, P, V)
    abs(L - prevloss) < loss_eps && Flux.stop()
    prevloss = L
    @printf("%12.7f\n", L)
  end
  opt = Flux.ADAM(lr)
  data = (
    random_minibatch(W, X, A, P, V; batchsize=batch_size)
    for i in 1:num_batches)
  print_legend()
  Flux.train!(loss, params(oracle.nn), data, opt, cb=Flux.throttle(cb, 1.0))
end

################################################################################

struct AlphaZero{Game, Board}
  params :: Params
  memory :: MemoryBuffer{Board}
  bestnn :: Oracle{Game}
  function AlphaZero{Game}(params) where Game
    Board = GI.Board(Game)
    memory = MemoryBuffer{Board}(params.mem_buffer_size)
    oracle = Oracle{Game}()
    new{Game, Board}(params, memory, oracle)
  end
end

################################################################################

struct MctsPlayer{M}
  mcts :: M
  niters :: Int
  τ :: Float64 # Temperature
  α :: Float64 # Dirichlet noise parameter
  ϵ :: Float64 # Dirichlet noise weight
  MctsPlayer(mcts, niters; τ=1., α=1., ϵ=0.) =
    new{typeof(mcts)}(mcts, niters, τ, α, ϵ)
end

function think(p::MctsPlayer, state)
  MCTS.explore!(p.mcts, state, p.niters)
  as, πa = MCTS.policy(p.mcts, τ=p.τ)
  if iszero(p.ϵ)
    return as, πa
  else
    noise = Dirichlet(length(πa), p.α)
    π = (1 - p.ϵ) * πa + p.ϵ * rand(noise)
    return as, π
  end
end

################################################################################

function self_play!(env::AlphaZero{Game}, p::MctsPlayer) where Game
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      push_game!(env.memory, z)
      return
    end
    actions, π = think(p, state)
    cboard = GI.canonical_board(state)
    push_sample!(env.memory, cboard, π, GI.white_playing(state))
    GI.play!(state, actions[rand(Categorical(π))])
  end
end

################################################################################
# Pitting Arena

function play_game(env::AlphaZero{Game}, white::MctsPlayer, black) where Game
  state = Game()
  while true
    z = GI.white_reward(state)
    isnothing(z) || (return z)
    player = GI.white_playing(state) ? white : black
    actions, π = think(player, state)
    GI.play!(state, actions[rand(Categorical(π))])
  end
end

# Returns average reward for the evaluated player
function evaluate(env::AlphaZero{G}, oracle) where G
  best_mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
  new_mcts = MCTS.Env{G}(oracle, env.params.cpuct)
  best = MctsPlayer(best_mcts, env.params.num_mcts_iters_per_turn, τ=TAU_EPS)
  new = MctsPlayer(new_mcts, env.params.num_mcts_iters_per_turn, τ=TAU_EPS)
  zsum = 0.
  best_first = true
  N = env.params.num_eval_games
  for i in 1:N
    white = best_first ? best : new
    black = best_first ? new : best
    z = play_game(env, white, black)
    best_first && (z = -z)
    zsum += z
    best_first = !best_first
  end
  return zsum / N
end

################################################################################

using ProgressMeter

function train!(env::AlphaZero{G}) where G
  # Collect data using self-play
  println("Collecting data using self-play....")
  mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
  player = MctsPlayer(mcts, env.params.num_mcts_iters_per_turn)
  @showprogress for i in 1:env.params.num_episodes_per_iter
    self_play!(env, player)
  end
  # Train new network
  newnn = copy(env.bestnn)
  examples = get(env.memory)
  println("Training new network.")
  train!(newnn, examples)
  z = evaluate(env, newnn)
  pwin = (z + 1) / 2
  @printf("Win rate of new network: %.0f%%\n", 100 * pwin)
  if pwin > env.params.update_threshold
    env.bestnn = newnn
    @printf("Replacing network.\n")
  end
end

################################################################################

function debug_tree(mcts; k=10)
  pairs = collect(mcts.tree)
  k = min(k, length(pairs))
  most_visited = sort(pairs, by=(x->x.second.Ntot), rev=true)[1:k]
  for (b, info) in most_visited
    println("N: ", info.Ntot)
    print_board(State(b))
  end
end

################################################################################
#=
include("tictactoe.jl")
env = AlphaZero{Game}(STD_PARAMS)

for i in 1:5
  train!(env)
end
=#
################################################################################
# Exploration tools

# Sort states by: who has guaranteed win, in how many states.
# How to print MCTS tree

################################################################################

const GAMES_DATA = "games.data"
using Serialization: serialize, deserialize

include("tictactoe.jl")

# Two steps: explore new network, explore dataset

#z = evaluate(env, newnn)
#pwin = (z + 1) / 2
#@printf("Win rate of new network: %.0f%%\n", 100 * pwin)

################################################################################

################################################################################
# Explore dataset

function inspect_memory(env, board::Board)
  mem = get(env.memory)
  relevant = findall(mem) do ex
    ex.b == board
  end
  N = length(relevant)
  iszero(N) && return nothing
  z = sum(mem[i].z for i in relevant) / N
  π = sum(mem[i].π for i in relevant) / N
  return N, z, π
end

function inspect_memory(env, state::State)
  b = GI.canonical_board(state)
  info = inspect_memory(env, b)
  if isnothing(info)
    println("Not in memory\n")
  else
    N, z, π = info
    @show N
    @show z
    println("")
    as = GI.available_actions(state)
    for (i, p) in sort(collect(enumerate(π)), by=(((i,p),)->p), rev=true)
      @printf("%1s %8.3f\n\n", action_str(as[i]), p)
    end
  end
end

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

function print_state_statistics(mcts, state)
  wp = GI.white_playing(state)
  b = GI.canonical_board(state)
  if haskey(mcts.tree, b)
    info = mcts.tree[b]
    @printf("N: %d, V: %.3f\n\n", info.Ntot, info.Vest)
    actions = enumerate(info.actions) |> collect
    actions = sort(actions, by=(((i,a),) -> info.stats[i].N), rev=true)
    ucts = MCTS.uct_scores(info, mcts.cpuct)
    @printf("%1s %7s %8s %6s %8s\n", "", "N (%)", "Q", "P", "UCT")
    for (i, a) in actions
      stats = info.stats[i]
      Nr = 100 * stats.N / info.Ntot
      Q = stats.N > 0 ? stats.W / stats.N : 0.
      astr = action_str(a)
      @printf("%1s %7.2f %8.4f %6.2f %8.4f\n", astr, Nr, Q, stats.P, ucts[i])
    end
  else
    print("Unexplored board.")
  end
  println("")
end

using DataStructures: Stack

mutable struct Explorer
  env
  state
  history
  mcts
  Explorer(env, state, mcts) = new(env, state, Stack{Any}(), mcts)
end

save_state!(exp::Explorer) = push!(exp.history, deepcopy(exp.state))

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
  end
  return false
end

function launch(exp::Explorer)
  while true
    # Print the state
    TicTacToe.print_board(exp.state, with_position_names=true)
    println("")
    print_state_statistics(exp.mcts, exp.state)
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

if !isfile(GAMES_DATA)
  env = AlphaZero{Game}(STD_PARAMS)
  mcts = MCTS.Env{Game}(env.bestnn, env.params.cpuct)
  player = MctsPlayer(mcts, 100)
  @showprogress for i in 1:100
    self_play!(env, player)
  end
  serialize(GAMES_DATA, (env, mcts))
else
  env, mcts = deserialize(GAMES_DATA)
end
println("Number of games collected: ", length(env.memory))


newnn = copy(env.bestnn)
examples = get(env.memory)
println("Training new network.")
train!(newnn, examples, num_batches = 100_000)


explorer = Explorer(env, State(), mcts)
launch(explorer)

#=
newnn = copy(env.bestnn)
examples = get(env.memory)
println("Training new network.")
train!(newnn, examples, num_batches = 100_000)
=#

#explorer = Explorer(env, State(), mcts)
#interpret!(explorer, "do", ["1e"])
#launch(explorer)

################################################################################
