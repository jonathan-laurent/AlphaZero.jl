################################################################################
# Setting up the network
################################################################################

using Printf
using AlphaZero.GameInterface
using AlphaZero.MCTS
using Distributions: Categorical, Dirichlet

################################################################################
# Params

using Base: @kwdef

@kwdef struct ArenaParams
  num_mcts_iters_per_turn :: Int = 25
  num_games :: Int = 40
  temperature :: Float64 = 0.4
  update_threshold :: Float64 = 0.55 # 0.6 in Nair'simplementation
end

@kwdef struct SelfPlayParams
  num_mcts_iters_per_turn :: Int = 250
  temperature :: Float64 = 1.
  dirichlet_noise_nα :: Float64 = 10.
  dirichlet_noise_ϵ :: Float64 = 0.25
end

@kwdef struct LearningParams
  learning_rate :: Float64 = 1e-3
  num_batches :: Int = 10_000
  batch_size :: Int = 32
  loss_eps :: Float64 = 1e-3
end

@kwdef struct Params
  arena :: ArenaParams = ArenaParams()
  self_play :: SelfPlayParams = SelfPlayParams()
  learning :: LearningParams = LearningParams()
  num_learning_iters :: Int = 100
  num_episodes_per_iter :: Int = 25
  mem_buffer_size :: Int = 200_000
  cpuct :: Float64 = 1.0
end

# Some standard values for params:
# https://github.com/suragnair/alpha-zero-general/blob/master/main.py
# For dirichlet noise, see:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3

################################################################################
# Alpha go test
# 4.9 million games of self play
# Parameters updated from 700,000 minibatches of 2048 positions
# Neural network: 20 residual blocks
# momenum param: 0.9
# Checkpoint after 1000 training steps
# First 30 moves, τ=1, then τ → 0
# Question: when pitting network against each other,
# where does randomness come from?
#=
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
=#
# Alpha go final:
# 29 million games, 31 million minibatches of 2048

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
  Flux.loadparams!(new.nn, Flux.params(o.nn))
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

using Statistics: mean
using DataStructures: CircularBuffer

struct TrainingExample{Board}
  b :: Board
  π :: Vector{Float64}
  z :: Float64
  n :: Int # Sample weight
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

Base.length(b::MemoryBuffer) = length(b.mem)

function merge_samples(es::Vector{TrainingExample{B}}) where B
  b = es[1].b
  π = mean(e.π for e in es)
  z = mean(e.z for e in es)
  n = sum(e.n for e in es)
  return TrainingExample{B}(b, π, z, n)
end

# Get memory content
function get(b::MemoryBuffer{B}) where B
  dict = Dict{B, Vector{TrainingExample{B}}}()
  sizehint!(dict, length(b))
  for e in b.mem[:]
    if haskey(dict, e.b)
      push!(dict[e.b], e)
    else
      dict[e.b] = [e]
    end
  end
  return [merge_samples(es) for es in values(dict)]
end

function push_sample!(buf::MemoryBuffer, board, policy, white_playing)
  player_code = white_playing ? 1.0 : -1.0
  ex = TrainingExample(board, policy, player_code, 1)
  push!(buf.cur, ex)
end

function push_game!(buf::MemoryBuffer, white_reward)
  for ex in buf.cur
    r = ex.z * white_reward
    push!(buf.mem, TrainingExample(ex.b, ex.π, r, ex.n))
  end
  empty!(buf.cur)
end

################################################################################

# concat_cols(cols) == hcat(cols...)
function concat_columns(cols)
  @assert !isempty(cols)
  nsamples = length(cols)
  excol = first(cols)
  sdim = length(excol)
  arr = similar(excol, (sdim, nsamples))
  for (i, col) in enumerate(cols)
    arr[:,i] = col
  end
  return arr
end

infinity(::Type{R}) where R <: Real = one(R) / zero(R)

weighted_mse(ŷ, y, w) = sum((ŷ .- y).^2 .* w) * 1 // length(y)

function random_minibatch(W, X, A, P, V; batchsize)
  n = size(X, 2)
  indices = rand(1:n, batchsize)
  return (W[:,indices], X[:,indices], A[:,indices], P[:,indices], V[:,indices])
end

function convert_sample(Game, e::TrainingExample)
  w = [log2(R(e.n)) + one(R)]
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
  W = concat_columns((e[1] for e in ces))
  X = concat_columns((e[2] for e in ces))
  A = concat_columns((e[3] for e in ces))
  P = concat_columns((e[4] for e in ces))
  V = concat_columns((e[5] for e in ces))
  return (W, X, A, P, V)
end

function train!(
    oracle::Oracle{G},
    examples::Vector{<:TrainingExample},
    params::LearningParams
  ) where G

  opt = Flux.ADAM(params.learning_rate)
  let (W, X, A, P, V) = convert_samples(G, examples)
  let prevloss = infinity(R)
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
    function cb()
      L = loss(W, X, A, P, V)
      prevloss - L < params.loss_eps && Flux.stop()
      prevloss = L
      @printf("%12.7f\n", L)
    end
    data = (
      random_minibatch(W, X, A, P, V; batchsize=params.batch_size)
      for i in 1:params.num_batches)
    print_legend()
    Flux.train!(
      loss, Flux.params(oracle.nn), data, opt, cb=Flux.throttle(cb, 1.0))
  end end
end

################################################################################

mutable struct AlphaZero{Game, Board}
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
  τ  :: Float64 # Temperature
  nα :: Float64 # Dirichlet noise parameter
  ϵ  :: Float64 # Dirichlet noise weight
  MctsPlayer(mcts, niters; τ=1., nα=10., ϵ=0.) =
    new{typeof(mcts)}(mcts, niters, τ, nα, ϵ)
end

function think(p::MctsPlayer, state)
  MCTS.explore!(p.mcts, state, p.niters)
  actions, π_mcts = MCTS.policy(p.mcts, τ=p.τ)
  if iszero(p.ϵ)
    π_exp = π_mcts
  else
    n = length(π_mcts)
    noise = Dirichlet(n, p.nα / n)
    π_exp = (1 - p.ϵ) * π_mcts + p.ϵ * rand(noise)
  end
  a = actions[rand(Categorical(π_exp))]
  return π_mcts, a
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
    π, a = think(p, state)
    cboard = GI.canonical_board(state)
    push_sample!(env.memory, cboard, π, GI.white_playing(state))
    GI.play!(state, a)
  end
end

################################################################################
# Pitting Arena

function play_game(env::AlphaZero{Game}, white::MctsPlayer, black) where Game
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z) return z :: Float64 end
    player = GI.white_playing(state) ? white : black
    π, a = think(player, state)
    GI.play!(state, a)
  end
end

# Returns average reward for the evaluated player
function evaluate_oracle(
    env::AlphaZero{G},
    oracle;
    τ = env.params.arena.temperature,
    num_mcts_iters = env.params.arena.num_mcts_iters_per_turn,
    num_games = env.params.arena.num_games
  ) where G
  best_mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
  new_mcts = MCTS.Env{G}(oracle, env.params.cpuct)
  best = MctsPlayer(best_mcts, num_mcts_iters, τ=τ)
  new = MctsPlayer(new_mcts, num_mcts_iters, τ=τ)
  zsum = 0.
  best_first = true
  for i in 1:num_games
    white = best_first ? best : new
    black = best_first ? new : best
    z = play_game(env, white, black)
    best_first && (z = -z)
    zsum += z
    best_first = !best_first
  end
  return zsum / num_games
end

################################################################################

using ProgressMeter

function train!(
    env::AlphaZero{G},
    num_iters=env.params.num_learning_iters
  ) where G
  mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
  for i in 1:num_iters
    # Collect data using self-play
    println("Collecting data using self-play....")
    player = MctsPlayer(mcts,
      env.params.self_play.num_mcts_iters_per_turn,
      τ = env.params.self_play.temperature,
      nα = env.params.self_play.dirichlet_noise_nα,
      ϵ = env.params.self_play.dirichlet_noise_ϵ)
    @showprogress for i in 1:env.params.num_episodes_per_iter
      self_play!(env, player)
    end
    # Train new network
    newnn = copy(env.bestnn)
    examples = get(env.memory)
    println("Training new network.")
    train!(newnn, examples, env.params.learning)
    z = evaluate_oracle(env, newnn)
    pwin = (z + 1) / 2
    @printf("Win rate of new network: %.0f%%\n", 100 * pwin)
    if pwin > env.params.arena.update_threshold
      env.bestnn = newnn
      mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
      @printf("Replacing network.\n")
    end
  end
end

################################################################################

include("tictactoe.jl")

################################################################################
# Explore dataset

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

using DataStructures: Stack

mutable struct Explorer
  env
  state
  history
  mcts
  oracle
  Explorer(env, state, mcts, oracle=nothing) =
    new(env, state, Stack{Any}(), mcts, oracle)
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
#=
using Serialization: serialize, deserialize

const GAMES_DATA = "games.data"

const CACHE = true

if !CACHE || !isfile(GAMES_DATA)
  env = AlphaZero{Game}(STD_PARAMS)
  mcts = MCTS.Env{Game}(env.bestnn, env.params.cpuct)
  player = MctsPlayer(mcts, 200, ϵ=0.25, nα=10.)
  @showprogress for i in 1:10_000
    self_play!(env, player)
  end
  serialize(GAMES_DATA, (env, mcts))
else
  env, mcts = deserialize(GAMES_DATA)
end
println("Number of games collected: ", length(env.memory))

show_memory_stats(env)

newnn = copy(env.bestnn)
examples = get(env.memory)
println("Training new network.")
train!(newnn, examples, num_batches=100_000)

z = evaluate(env, newnn, num_mcts_iters=10, num_games=10)
pwin = (z + 1) / 2
@printf("Win rate of new network: %.0f%%\n", 100 * pwin)

explorer = Explorer(env, State(), mcts, newnn)
launch(explorer)
=#

################################################################################

arena = ArenaParams(
  update_threshold=0.55,
  num_mcts_iters_per_turn=20,
  num_games=100)

self_play = SelfPlayParams(
  num_mcts_iters_per_turn=100)

learning = LearningParams()

params = Params(self_play=self_play, arena=arena,
  num_learning_iters=3,
  num_episodes_per_iter=1000)

env = AlphaZero{Game}(params)

train!(env)

################################################################################
