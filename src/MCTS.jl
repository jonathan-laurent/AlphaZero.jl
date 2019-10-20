"""
    MCTS

A generic implementation of Monte Carlo Tree Search (with an external oracle).
Relies on `AlphaZero.GameInterface`.

"""
module MCTS

using DataStructures: Stack
using Distributions: Categorical

using ..Util: @printing_errors
import ..GI

#####
##### Interface for external oracles
#####

abstract type Oracle{Game} end

function evaluate end

# The simplest way to evaluate a position is to perform rollouts
# Alternatively, the user can provide a NN-based oracle
struct RolloutOracle{Game} <: Oracle{Game} end

function rollout(::Type{Game}, board) where Game
  state = Game(board)
  while true
    reward = GI.white_reward(state)
    isnothing(reward) || (return reward)
    action = rand(GI.available_actions(state))
    GI.play!(state, action)
   end
end

function evaluate(::RolloutOracle{G}, board, available_actions) where G
  V = rollout(G, board)
  n = length(available_actions)
  P = [1 / n for a in available_actions]
  return P, V
end

# Default implementation (inefficient)
function evaluate_batch(oracle::Oracle, requests)
  return [evaluate(oracle, b, a) for (b, a) in requests]
end

#####
##### MCTS Environment
#####

struct ActionStats
  P :: Float32
  W :: Float64
  N :: Int
  nworkers :: UInt8 # Number of workers currently exploring this branch
end

struct BoardInfo
  stats :: Vector{ActionStats}
  Vest  :: Float32
end

set_nworkers(a::ActionStats, n) = ActionStats(a.P, a.W, a.N, n)

Ntot(b::BoardInfo) = sum(s.N for s in b.stats)

symmetric_reward(r) = -r

const InferenceRequest{B, A} = Union{Nothing, Tuple{B, Vector{A}}}

const InferenceResult{R} = Tuple{Vector{R}, R}

mutable struct Worker{Game, Board, Action}
  id    :: Int
  tree  :: Dict{Board, BoardInfo} # Shared reference to the tree
  stack :: Stack{Tuple{Board, Bool, Int}} # board, white_playing, action_number
  cpuct :: Float64
  send  :: Channel{InferenceRequest{Board, Action}}
  recv  :: Channel{InferenceResult{Float32}}
  global_lock :: ReentrantLock
  
  function Worker{G, B, A}(id, tree, lock, cpuct) where {G, B, A}
    stack = Stack{Tuple{B, Bool, Int}}()
    send = Channel{InferenceRequest{B, A}}(1)
    recv = Channel{InferenceResult{Float32}}(1)
    new{G, B, A}(id, tree, stack, cpuct, send, recv, lock)
  end
end

mutable struct Env{Game, Board, Action, Oracle}
  # Store state statistics assuming player one is to play for nonterminal states
  tree :: Dict{Board, BoardInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Performance statistics
  total_time :: Float64
  inference_time :: Float64
  # Workers
  workers :: Vector{Worker{Game, Board, Action}}
  
  function Env{G}(oracle, nworkers, cpuct=1.) where G
    B = GI.Board(G)
    A = GI.Action(G)
    tree = Dict{B, BoardInfo}()
    total_time, inference_time = 0., 0.
    glock= ReentrantLock()
    workers = [Worker{G, B, A}(i, tree, glock, cpuct) for i in 1:nworkers]
    new{G, B, A, typeof(oracle)}(
      tree, oracle, total_time, inference_time, workers)
  end
end

#####
##### Access and initialize state information
#####

# Returns statistics for the current player, true if new node
function board_info(worker, board, actions)
  #@info "Worker $(worker.id) sees a tree of size $(length(worker.tree))"
  if haskey(worker.tree, board)
    return (worker.tree[board], false)
  else
    #@info "Worker $(worker.id) launches a query for $(hash(board)%100)."
    # Send a request to the inference server
    put!(worker.send, (board, actions))
    unlock(worker.global_lock)
    P, V = take!(worker.recv)
    lock(worker.global_lock)
    # Another worker may have sent the same request and initialized
    # the node before. Therefore, we have to test membership again.
    if !haskey(worker.tree, board)
      #@info "Worker $(worker.id) initializes node for $(hash(board)%100)."
      stats = [ActionStats(p, 0, 0, 0) for p in P]
      info = BoardInfo(stats, V)
      worker.tree[board] = info
      return (info, true)
    else
      # The inference result is ignored and we proceed as if
      # the node was already in the tree.
      #@info "Worker $(worker.id) ignores query result for board $(hash(board)%100)."
      return (worker.tree[board], false)
    end
  end
end

#####
##### Exploration utilities
#####

function debug_tree(env::Env{Game}; k=10) where Game
  pairs = collect(env.tree)
  k = min(k, length(pairs))
  most_visited = sort(pairs, by=(x->Ntot(x.second)), rev=true)[1:k]
  for (b, info) in most_visited
    println("N: ", Ntot(info))
    GI.print_state(Game(b))
  end
end

#####
##### Main algorithm
#####

function uct_scores(info::BoardInfo, cpuct)
  sqrtNtot = sqrt(Ntot(info))
  return map(info.stats) do a
    Q = (a.W - a.nworkers) / max(a.N, 1)
    Q + cpuct * a.P * sqrtNtot / (a.N + 1)
  end
end

function push_board_action!(worker, (b, wp, aid))
  push!(worker.stack, (b, wp, aid))
  #@info "Worker $(worker.id) increments vloss for board $(hash(b)%100), action $(aid)"
  stats = worker.tree[b].stats
  stats[aid] = set_nworkers(stats[aid], stats[aid].nworkers + 1)
end

function select!(worker, state)
  #@info "Worker $(worker.id) starts selection process"
  state = copy(state)
  while true
    wr = GI.white_reward(state)
    isnothing(wr) || (return wr)
    wp = GI.white_playing(state)
    board = GI.canonical_board(state)
    actions = GI.available_actions(state)
    let (info, new_node) = board_info(worker, board, actions)
      new_node && (return info.Vest)
      scores = uct_scores(info, worker.cpuct)
      #@info "Worker $(worker.id) sees ws: $([a.W for a in info.stats])"
      #@info "Worker $(worker.id) sees nworkers: $([a.nworkers for a in info.stats])"
      #@info "Worker $(worker.id) sees uct scores: $(scores)"
      best_action_id = argmax(scores)
      best_action = actions[best_action_id]
      push_board_action!(worker, (board, wp, best_action_id))
      GI.play!(state, best_action)
    end
  end
end

function backprop!(worker, white_reward)
  while !isempty(worker.stack)
    board, white_playing, action_id = pop!(worker.stack)
    reward = white_playing ?
      white_reward :
      symmetric_reward(white_reward)
    stats = worker.tree[board].stats
    astats = stats[action_id]
    #@info "Worker $(worker.id) decrements vloss for board $(hash(board)%100) (action $(action_id))"
    stats[action_id] = ActionStats(
      astats.P, astats.W + reward, astats.N + 1, astats.nworkers - 1)
  end
end

function explore!(worker::Worker, state, nsims)
  lock(worker.global_lock)
  for i in 1:nsims
    @assert isempty(worker.stack)
    white_reward = select!(worker, state)
    backprop!(worker, white_reward)
    @assert isempty(worker.stack)
  end
  put!(worker.send, nothing) # send termination message to the server
  unlock(worker.global_lock)
end

# Does not evaluate finite number of batches
function inference_server(env::Env{G, B, A}) where {G, B, A}
  to_watch = env.workers
  while true
    #@info "Server waiting for requests..."
    requests = [take!(w.send) for w in to_watch]
    active = [!isnothing(r) for r in requests]
    any(active) || break
    batch = convert(Vector{Tuple{B, Vector{A}}}, requests[active])
    to_watch = to_watch[active]
    @assert !isempty(batch)
    answers, time = @timed evaluate_batch(env.oracle, batch)
    env.inference_time += time
    for i in eachindex(batch)
      #@info "Sending answer to worker $(to_watch[i].id)"
      put!(to_watch[i].recv, answers[i])
    end
  end
end

function explore!(env::Env, state, nsims=1)
  # Amount of work per worker
  # (rounding nsims to the upper multiple of nworkers)
  pw = ceil(Int, nsims / length(env.workers))
  elapsed = @elapsed begin
    @sync begin
      #@info "Launching server"
      @async @printing_errors inference_server(env)
      for w in env.workers
        #@info "Launching worker $(w.id)"
        @async @printing_errors explore!(w, state, pw)
      end
    end
  end
  env.total_time += elapsed
  return
end

# Returns (actions, distr)
function policy(env::Env, state; τ=1.0)
  actions = GI.available_actions(state)
  board = GI.canonical_board(state)
  info =
    try env.tree[board]
    catch e
      if isa(e, KeyError)
        error("MCTS.explore! must be called before MCTS.policy")
      else
        rethrow(e)
      end
    end
  τinv = 1 / τ
  D = [a.N ^ τinv for a in info.stats]
  return actions, D ./ sum(D)
end

function reset!(env)
  empty!(env.tree)
  return
end

function reset!(env, oracle)
  reset!(env)
  env.oracle = oracle
  return
end

function inference_time_ratio(env)
  T = env.total_time
  iszero(T) ? 0. : env.inference_time / T
end

#####
##### MCTS AI (for illustration purposes)
#####

struct AI <: GI.Player
  env :: Env
  step :: Int
  timeout :: Float64
  random :: Bool
  function AI(env; step=1024, timeout=3., random=false)
    new(env, step, timeout, random)
  end
end

function GI.select_move(ai::AI, state)
  start = time()
  while time() - start < ai.timeout
    explore!(ai.env, state, ai.step)
  end
  actions, distr = policy(ai.env, state)
  if ai.random
    return actions[rand(Categorical(distr))]
  else
    return actions[argmax(distr)]
  end
end

end
