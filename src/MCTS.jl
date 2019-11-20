"""
A generic, standalone implementation of asynchronous Monte Carlo Tree Search.
It can be used on any game that implements the [`GameInterface`](@ref
Main.AlphaZero.GameInterface) interface and with any external oracle.
"""
module MCTS

using DataStructures: Stack
using Distributions: Categorical

using ..Util: @printing_errors
import ..GI

#####
##### Interface for external oracles
#####

"""
    MCTS.Oracle{Game}

Abstract base type for an oracle. Oracles must implement
[`MCTS.evaluate`](@ref) and [`MCTS.evaluate_batch`](@ref).
"""
abstract type Oracle{Game} end

"""
    MCTS.evaluate(oracle::Oracle, board, available_actions)

Evaluate a single board position (assuming white is playing).

Return a pair `(P, V)` where:

  - `P` is a probability vector on available actions
  - `V` is a scalar estimating the value or win probability for white.
"""
function evaluate end

"""
    MCTS.evaluate_batch(oracle::Oracle, batch)

Evaluate a batch of board positions.

Expect a vector of `(board, available_actions)` pairs and
return a vector of `(P, V)` pairs.

A default implementation is provided that calls [`MCTS.evaluate`](@ref)
sequentially on each position.
"""
function evaluate_batch(oracle::Oracle, batch)
  return [evaluate(oracle, b, a) for (b, a) in requests]
end

"""
    MCTS.RolloutOracle{Game} <: MCTS.Oracle{Game}

This oracle estimates the value of a position by simulating a random game
from it (a rollout). Moreover, it puts a uniform prior on available actions.
Therefore, it can be used to implement the "vanilla" MCTS algorithm.
"""
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

struct RandomOracle{Game} <: Oracle{Game} end

function evaluate(::RandomOracle, board, actions)
  n = length(actions)
  P = ones(n) ./ n
  V = 0.
  return P, V
end

#####
##### MCTS Environment
#####

struct ActionStats
  P :: Float32
  W :: Float64
  N :: Int
  nworkers :: UInt16 # Number of workers currently exploring this branch
end

struct BoardInfo
  stats :: Vector{ActionStats}
  Vest  :: Float32
end

Ntot(b::BoardInfo) = sum(s.N for s in b.stats)

symmetric_reward(r::Real) = -r

const InferenceRequest{B, A} = Union{Nothing, Tuple{B, Vector{A}}}

const InferenceResult{R} = Tuple{Vector{R}, R}

mutable struct Worker{Board, Action}
  id    :: Int # Useful for debugging purposes
  stack :: Stack{Tuple{Board, Bool, Int}} # board, white_playing, action_number
  send  :: Channel{InferenceRequest{Board, Action}}
  recv  :: Channel{InferenceResult{Float32}}

  function Worker{B, A}(id) where {B, A}
    stack = Stack{Tuple{B, Bool, Int}}()
    send = Channel{InferenceRequest{B, A}}(1)
    recv = Channel{InferenceResult{Float32}}(1)
    new{B, A}(id, stack, send, recv)
  end
end

"""
    MCTS.Env{Game}(oracle; nworkers=1, fill_batches=false, cpuct=1.) where Game

Create and initialize an MCTS environment.

## Arguments

  - `oracle`: external oracle
  - `nworkers`: numbers of asynchronous workers (see below)
  - `fill_batches`: if true, a constant batch size is enforced for evaluation
     requests, by completing batches with dummy entries if necessary
  - `cpuct`: exploration constant (in the UCT formula)

## Asynchronous MCTS

  - If `nworkers == 1`, MCTS is run in a synchronous fashion and the oracle is
    invoked through [`MCTS.evaluate`](@ref).

  - If `nworkers > 1`, `nworkers` asynchronous workers are spawned,
    along with an additional task to serve board evaluation requests.
    Such requests are processed by batches of
    size `nworkers` using [`MCTS.evaluate_batch`](@ref).
"""
mutable struct Env{Game, Board, Action, Oracle}
  # Store (nonterminal) state statistics assuming player one is to play
  tree :: Dict{Board, BoardInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Workers
  workers :: Vector{Worker{Board, Action}}
  global_lock :: ReentrantLock
  remaining :: Int # Iterations counters
  # Parameters
  fill_batches :: Bool
  cpuct :: Float64
  # Performance statistics
  total_time :: Float64
  inference_time :: Float64
  total_iterations :: Int64
  total_nodes_traversed :: Int64

  function Env{G}(oracle; nworkers=1, fill_batches=false, cpuct=1.) where G
    B = GI.Board(G)
    A = GI.Action(G)
    tree = Dict{B, BoardInfo}()
    total_time = 0.
    inference_time = 0.
    total_iterations = 0
    total_nodes_traversed = 0
    lock = ReentrantLock()
    remaining = 0
    workers = [Worker{B, A}(i) for i in 1:nworkers]
    new{G, B, A, typeof(oracle)}(
      tree, oracle, workers, lock, remaining, fill_batches, cpuct,
      total_time, inference_time, total_iterations, total_nodes_traversed)
  end
end

"""
    MCTS.memory_footprint_per_node(env)

Return an estimate of the memory footprint of a single node
of the MCTS tree (in bytes).
"""
function memory_footprint_per_node(env::Env{G}) where G
  # The hashtable is at most twice the number of stored elements
  # For every element, a board and a pointer are stored
  size_key = 2 * (GI.board_memsize(G) + sizeof(Int))
  dummy_stats = BoardInfo([
    ActionStats(0, 0, 0, 0) for i in 1:GI.num_actions(G)], 0)
  size_stats = Base.summarysize(dummy_stats)
  return size_key + size_stats
end

# Possibly very slow for large trees
memory_footprint(env::Env) = Base.summarysize(env.tree)

"""
    MCTS.approximate_memory_footprint(env)

Return an estimate of the memory footprint of the MCTS tree (in bytes).
"""
function approximate_memory_footprint(env::Env)
  return memory_footprint_per_node(env) * length(env.tree)
end

asynchronous(env::Env) = length(env.workers) > 1

#####
##### Access and initialize state information
#####

function init_board_info(P, V)
  stats = [ActionStats(p, 0, 0, 0) for p in P]
  return BoardInfo(stats, V)
end

# Returns statistics for the current player, true if new node
# Synchronous version
function board_info_sync(env, worker, board, actions)
  if haskey(env.tree, board)
    return (env.tree[board], false)
  else
    (P, V), time = @timed evaluate(env.oracle, board, actions)
    env.inference_time += time
    info = init_board_info(P, V)
    env.tree[board] = info
    return (info, true)
  end
end

function board_info_async(env, worker, board, actions)
  if haskey(env.tree, board)
    return (env.tree[board], false)
  else
    # Send a request to the inference server
    put!(worker.send, (board, actions))
    unlock(env.global_lock)
    P, V = take!(worker.recv)
    lock(env.global_lock)
    # Another worker may have sent the same request and initialized
    # the node before. Therefore, we have to test membership again.
    if !haskey(env.tree, board)
      info = init_board_info(P, V)
      env.tree[board] = info
      return (info, true)
    else
      # The inference result is ignored and we proceed as if
      # the node was already in the tree.
      return (env.tree[board], false)
    end
  end
end

function board_info(env, worker, board, actions)
  asynchronous(env) ?
    board_info_async(env, worker, board, actions) :
    board_info_sync(env, worker, board, actions)
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

function push_board_action!(env, worker, (b, wp, aid))
  push!(worker.stack, (b, wp, aid))
  stats = env.tree[b].stats
  astats = stats[aid]
  stats[aid] = ActionStats(
    astats.P, astats.W, astats.N + 1, astats.nworkers + 1)
end

function select!(env, worker, state)
  state = copy(state)
  env.total_iterations += 1
  while true
    wr = GI.white_reward(state)
    isnothing(wr) || (return wr)
    wp = GI.white_playing(state)
    board = GI.canonical_board(state)
    actions = GI.available_actions(state)
    let (info, new_node) = board_info(env, worker, board, actions)
      new_node && (return info.Vest)
      scores = uct_scores(info, env.cpuct)
      best_action_id = argmax(scores)
      best_action = actions[best_action_id]
      push_board_action!(env, worker, (board, wp, best_action_id))
      GI.play!(state, best_action)
      env.total_nodes_traversed += 1
    end
  end
end

function backprop!(env, worker, white_reward)
  while !isempty(worker.stack)
    board, white_playing, action_id = pop!(worker.stack)
    reward = white_playing ?
      white_reward :
      symmetric_reward(white_reward)
    stats = env.tree[board].stats
    astats = stats[action_id]
    stats[action_id] = ActionStats(
      astats.P, astats.W + reward, astats.N, astats.nworkers - 1)
  end
end

function worker_explore!(env::Env, worker::Worker, state)
  @assert isempty(worker.stack)
  white_reward = select!(env, worker, state)
  backprop!(env, worker, white_reward)
  @assert isempty(worker.stack)
end

# Does not evaluate finite number of batches
function inference_server(env::Env{G, B, A}) where {G, B, A}
  to_watch = env.workers
  while true
    requests = [take!(w.send) for w in to_watch]
    active = [!isnothing(r) for r in requests]
    n_active = count(active)
    n_active > 0 || break
    batch = convert(Vector{Tuple{B, Vector{A}}}, requests[active])
    to_watch = to_watch[active]
    @assert !isempty(batch)
    if env.fill_batches
      nmissing = length(env.workers) - length(batch)
      if nmissing > 0
        append!(batch, [batch[1] for i in 1:nmissing])
      end
      @assert length(batch) == length(env.workers)
    end
    answers, time = @timed evaluate_batch(env.oracle, batch)
    env.inference_time += time
    for i in 1:n_active
      put!(to_watch[i].recv, answers[i])
    end
  end
end

function explore_sync!(env::Env, state, nsims)
  elapsed = @elapsed for i in 1:nsims
    worker_explore!(env, env.workers[1], state)
  end
  env.total_time += elapsed
end

function explore_async!(env::Env, state, nsims)
  env.remaining = nsims
  elapsed = @elapsed begin
    @sync begin
      @async @printing_errors inference_server(env)
      for w in env.workers
        @async @printing_errors begin
          lock(env.global_lock)
          while env.remaining > 0
            env.remaining -= 1
            worker_explore!(env, w, state)
          end
          put!(w.send, nothing) # send termination message to the server
          unlock(env.global_lock)
        end
      end
    end
  end
  env.total_time += elapsed
  return
end

"""
    MCTS.explore!(env, state, nsims)

Run `nsims` MCTS iterations from `state`.

In case there are multiple workers, `nsims` is rounded up to the nearest
integer multiple of the number of workers.
"""
function explore!(env::Env, state, nsims)
  asynchronous(env) ?
    explore_async!(env, state, nsims) :
    explore_sync!(env, state, nsims)
end

"""
    MCTS.policy(env, state; τ=1.)

Return the recommended stochastic policy on `state`,
with temperature parameter equal to `τ`.

A call to this function must always be preceded by
a call to [`MCTS.explore!`](@ref).
"""
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

"""
    MCTS.reset!(env)

Empty the MCTS tree.
"""
function reset!(env)
  empty!(env.tree)
end

"""
    MCTS.inference_time_ratio(env)

Return the ratio of time spent by [`MCTS.explore!`](@ref)
on position evaluation (through functions [`MCTS.evaluate`](@ref) or
[`MCTS.evaluate_batch`](@ref)) since the environment's creation.
"""
function inference_time_ratio(env)
  T = env.total_time
  iszero(T) ? 0. : env.inference_time / T
end

"""
    MCTS.average_exploration_depth(env)

Return the average number of nodes that are traversed during an
MCTS iteration, not counting the root.
"""
function average_exploration_depth(env)
  return env.total_nodes_traversed / env.total_iterations
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
