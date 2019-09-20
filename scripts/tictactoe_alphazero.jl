################################################################################
# Setting up the network
################################################################################

using AlphaZero.GameInterface
using AlphaZero.MCTS
using Distributions: Categorical

################################################################################
# Params

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

const STD_PARAMS = Params(;
  num_learning_iters       = 1_000,
  num_episodes_per_iter    = 100,
  num_mcts_iters_per_turn  = 50,  # 25 in Nair's implementation
  mem_buffer_size          = 200_000,
  num_eval_games           = 40,
  update_threshold         = 0.6,
  dirichlet_noise_n        = 10,
  cpuct                    = 1.0)

# Alpha go test
# 4.9 million games of self play
# Parameters updated from 700,000 minibatches of 2048 positions
# Neural network: 20 residual blocks
# momenum para: 0.9
# c = 1e-4
# Checkpoint after 1000 training steps
# First 30 moves, τ=1, then τ → 0
# Question: when pitting network against each other,
# where does randomness come from?
const ALPHA_GO_ZERO_PARAMS = Params(;
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

struct Oracle{Game}
  nn :: Network

  function Oracle{G}() where G
    hsize = 100
    nn = Network(GI.board_dim(G), GI.num_actions(G), hsize)
    nparams = sum(length(p) for p in params(nn))
    println("Number of parameters: ", nparams)
    new(nn)
  end
end

function MCTS.evaluate(ev::Oracle{G}, board, available_actions) where G
  # Build the mask
  nactions = GI.num_actions(G)
  mask = falses(nactions)
  for a in available_actions
    mask[GI.action_id(G, a)] = true
  end
  # Call the NN
  input = GI.vectorize_board(G, board)
  P, V = ev.nn(input, mask)
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

function push_state!(buf::MemoryBuffer, board, policy, white_playing)
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

struct AlphaZero{Game, Board, Action}
  params :: Params
  memory :: MemoryBuffer{Board}
  bestnn :: Oracle{Game}
  function AlphaZero{G, B, A}(params) where {G, B, A}
    memory = MemoryBuffer{B}(params.mem_buffer_size)
    new(params, memory)
  end
end

AlphaZero(Game) = AlphaZero{Game, GI.Board(Game), GI.Action(Game)}

################################################################################

function play_game!(env::AlphaZero{Game}, mcts; record=true) where Game
  state = Game()
  while true
    z = GI.white_reward(state)
    if !isnothing(z)
      record && push_game!(env.memory, z)
      return z
    end
    explore!(mcts, state, env.params.num_mcts_iters_per_turn)
    actions, π = policy(mcts)
    record && push_state!(
      env.memory, GI.board(state), π, GI.white_playing(state))
    GI.play!(state, actions[rand(Categorical(π))])
  end
end

################################################################################
# Pitting Arena.
# There are two ways to fight: build an oracle that behaves
# differently for white and black OR build two separate players

#=
using Statistics: mean

function evaluate(env::AlphaZero, oracle)
  N = 1:env.params.num_eval_games
  mcts = ...
  z = mean(play_game(env, record=false) for i in 1:N)
end
=#
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
env = AlphaZero(Game)(STD_PARAMS)
oracle = Oracle{Game}()
mcts = MCTS.Env{Game, GI.Board(Game), GI.Action(Game)}(oracle)
play_game!(env, mcts; record=true)
debug_tree(mcts)

################################################################################
