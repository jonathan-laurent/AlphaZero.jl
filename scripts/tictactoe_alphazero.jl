################################################################################
# Setting up the network
################################################################################

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
  cpuct                    :: Float64
end

Records.generate_named_constructors(Params) |> eval

# Some standard values for params:
# https://github.com/suragnair/alpha-zero-general/blob/master/main.py
# For dirichlet noise, see:
# https://medium.com/oracledevs/lessons-from-alphazero-part-3

const STD_PARAMS = Params(
  num_learning_iters       = 1_000,
  num_episodes_per_iter    = 100,  # 100
  num_mcts_iters_per_turn  = 100,  # 25 in Nair's implementation
  mem_buffer_size          = 200_000,
  num_eval_games           = 40,
  update_threshold         = 0.6,
  dirichlet_noise_n        = 10,
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
  cpuct                    = 1.0)

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

function push_sample!(buf::MemoryBuffer, board, policy, white_playing)
  player_code = white_playing ? 1.0 : -1.0
  ex = TrainingExample(board, policy, player_code)
  push!(buf.cur, ex)
end

function push_game!(buf::MemoryBuffer, white_reward)
  for ex in buf.cur
    r = ex.z * white_reward
    push!(buf.mem, TrainingExample(ex.b, ex.π, r))
  end
  empty!(buf.cur)
end

################################################################################

const BATCH_SIZE = 32
const NUM_GRADIENT_STEPS = 1000
const LR = 1e-3

norm2(x) = x' * x

function random_minibatch(X, A, P, V; batchsize)
  n = size(X, 2)
  indices = rand(1:n, batchsize)
  return ((X[:,indices], A[:,indices]), (P[:,indices], V[:,indices]))
end

using Printf

function convert_sample(Game, e::TrainingExample)
  x = Vector{R}(GI.vectorize_board(Game, e.b))
  actions = GI.available_actions(Game(e.b))
  a = GI.actions_mask(Game, actions)
  p = zeros(R, size(a))
  p[[GI.action_id(Game, a) for a in actions]] = e.π
  v = [R(e.z)]
  return (x, a, p, v)
end

function train!(oracle::Oracle{G}, examples::Vector{<:TrainingExample}) where G
  ces = [convert_sample(G, e) for e in examples]
  X = reduce(hcat, (e[1] for e in ces))
  A = reduce(hcat, (e[2] for e in ces))
  P = reduce(hcat, (e[3] for e in ces))
  V = reduce(hcat, (e[4] for e in ces))

  print_legend() = @printf("%12s", "Loss\n")

  function loss((X, A), (P₀, V₀))
    P, V = oracle.nn(X, A)
    return Flux.crossentropy(P .+ eps(R), P₀) + Flux.mse(V, V₀)
  end

  function cb()
    L = loss((X, A), (P, V))
    @printf("%12.7f\n", L)
  end

  opt = Flux.ADAM(LR)
  data = (
    random_minibatch(X, A, P, V; batchsize=BATCH_SIZE)
    for i in 1:NUM_GRADIENT_STEPS)
  @show loss((X, A), (P, V))
  #Flux.train!(loss, params(oracle.nn), data, opt, cb=Flux.throttle(cb, 5))
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
end

function think(p::MctsPlayer, state)
  MCTS.explore!(p.mcts, state, p.niters)
  MCTS.policy(p.mcts)
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
    push_sample!(env.memory, GI.board(state), π, GI.white_playing(state))
    GI.play!(state, actions[rand(Categorical(π))])
  end
end

################################################################################
# Pitting Arena.

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
  best = MctsPlayer(best_mcts, env.params.num_mcts_iters_per_turn)
  new = MctsPlyer(new_mcts, env.params.num_mcts_iters_per_turn)
  zsum = 0.0
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
  @progress for i in 1:env.params.num_episodes_per_iter
    self_play!(env, player)
  end
  # Train new network
  newnn = copy(env.bestnn)
  examples = get(env.memory)
  train!(newnn, examples)
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

include("tictactoe.jl")
env = AlphaZero{Game}(STD_PARAMS)
train!(env)

################################################################################
