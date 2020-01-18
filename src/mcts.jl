"""
A generic, standalone implementation of asynchronous Monte Carlo Tree Search.
It can be used on any game that implements the `GameInterface`
interface and with any external oracle.
"""
module MCTS

using DataStructures: Stack
using Distributions: Categorical, Dirichlet

using ..Util: @printing_errors, @unimplemented
import ..GI, ..GameType

#####
##### Interface for External Oracles
#####

"""
    MCTS.Oracle{Game}

Abstract base type for an oracle. Oracles must implement
[`MCTS.evaluate`](@ref) and [`MCTS.evaluate_batch`](@ref).
"""
abstract type Oracle{Game} end

"""
    MCTS.evaluate(oracle::Oracle, board)

Evaluate a single board position (assuming white is playing).

Return a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(Game(board))`
  - `V` is a scalar estimating the value or win probability for white.
"""
function evaluate(oracle::Oracle, board)
  @unimplemented
end

"""
    MCTS.evaluate_batch(oracle::Oracle, boards)

Evaluate a batch of board positions.

Expect a vector of boards and return a vector of `(P, V)` pairs.

A default implementation is provided that calls [`MCTS.evaluate`](@ref)
sequentially on each position.
"""
function evaluate_batch(oracle::Oracle, boards)
  return [evaluate(oracle, b) for b in boards]
end

GameType(::Oracle{Game}) where Game = Game

#####
##### Some Example Oracles
#####

"""
    MCTS.RolloutOracle{Game} <: MCTS.Oracle{Game}

This oracle estimates the value of a position by simulating a random game
from it (a rollout). Moreover, it puts a uniform prior on available actions.
Therefore, it can be used to implement the "vanilla" MCTS algorithm.
"""
struct RolloutOracle{Game} <: Oracle{Game} end

function rollout!(state, board)
  while true
    reward = GI.white_reward(state)
    isnothing(reward) || (return reward)
    action = rand(GI.available_actions(state))
    GI.play!(state, action)
   end
end

function evaluate(::RolloutOracle{Game}, board) where Game
  state = Game(board)
  n = length(GI.available_actions(state))
  P = ones(n) ./ n
  V = rollout!(state, board)
  return P, V
end

struct RandomOracle{Game} <: Oracle{Game} end

function evaluate(::RandomOracle{Game}, board) where Game
  s = Game(board)
  n = length(GI.available_actions(s))
  P = ones(n) ./ n
  V = 0.
  return P, V
end

#####
##### Workers and Queries
#####

# Workers can send three types of messages to the inference server:
# - Done: there is no more work available for the worker
# - Query: the worker requests the server to evaluate a position
# - NoQuery: the worker finished an MCTS simulation without sending
#   an evaluation query (see `worker_yield!` for why this matters)

abstract type Message{B} end
struct Done{B} <: Message{B} end
struct NoQuery{B} <: Message{B} end
struct Query{B} <: Message{B} board :: B end
const AnyMessage{B} = Union{Done{B}, NoQuery{B}, Query{B}}
const EvaluationResult{R} = Tuple{Vector{R}, R}

mutable struct Worker{Board}
  id    :: Int # useful for debugging purposes
  stack :: Stack{Tuple{Board, Bool, Int}} # board, white_playing, action_number
  send  :: Channel{AnyMessage{Board}}
  recv  :: Channel{EvaluationResult{Float32}}
  queried :: Bool # the worker queried the server during the current simulation

  function Worker{B}(id) where B
    stack = Stack{Tuple{B, Bool, Int}}()
    send = Channel{AnyMessage{B}}(1)
    recv = Channel{EvaluationResult{Float32}}(1)
    new{B}(id, stack, send, recv, false)
  end
end

#####
##### Board Statistics
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

#####
##### MCTS Environment
#####

"""
    MCTS.Env{Game}(oracle; <keyword args>) where Game

Create and initialize an MCTS environment with a given `oracle`.

## Keyword Arguments

  - `nworkers=1`: numbers of asynchronous workers (see below)
  - `fill_batches=false`: if true, a constant batch size is enforced for
     evaluation requests, by completing batches with dummy entries if necessary
  - `cpuct=1.`: exploration constant in the UCT formula
  - `noise_ϵ=0., noise_α=1.`: parameters for the dirichlet exploration noise
     (see below)

## Asynchronous MCTS

  - If `nworkers == 1`, MCTS is run in a synchronous fashion and the oracle is
    invoked through [`MCTS.evaluate`](@ref).

  - If `nworkers > 1`, `nworkers` asynchronous workers are spawned,
    along with an additional task to serve board evaluation requests.
    Such requests are processed by batches of
    size `nworkers` using [`MCTS.evaluate_batch`](@ref).

## Dirichlet Noise

A naive way to ensure exploration during training is to adopt an ϵ-greedy
policy, playing a random move at every turn instead of using the policy
prescribed by [`MCTS.policy`](@ref) with probability ϵ.
The problem with this naive strategy is that it may lead the player to make
terrible moves at critical moments, thereby biasing the policy evaluation
mechanism.

A superior alternative is to add a random bias to the neural prior for the root
node during MCTS exploration: instead of considering the policy ``p`` output
by the neural network in the UCT formula, one uses ``(1-ϵ)p + ϵη`` where ``η``
is drawn once per call to [`MCTS.explore!`](@ref) from a Dirichlet distribution
of parameter ``α``.
"""
mutable struct Env{Game, Board, Oracle}
  # Store (nonterminal) state statistics assuming player one is to play
  tree :: Dict{Board, BoardInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Workers
  workers :: Vector{Worker{Board}}
  global_lock :: ReentrantLock
  remaining :: Int # counts the number of remaining simulations to do
  # Parameters
  fill_batches :: Bool
  cpuct :: Float64
  noise_ϵ :: Float64
  noise_α :: Float64
  # Performance statistics
  total_time :: Float64
  inference_time :: Float64
  total_simulations :: Int64
  total_nodes_traversed :: Int64

  function Env{G}(oracle;
      nworkers=1, fill_batches=false,
      cpuct=1., noise_ϵ=0., noise_α=1.) where G
    B = GI.Board(G)
    tree = Dict{B, BoardInfo}()
    total_time = 0.
    inference_time = 0.
    total_simulations = 0
    total_nodes_traversed = 0
    lock = ReentrantLock()
    remaining = 0
    workers = [Worker{B}(i) for i in 1:nworkers]
    new{G, B, typeof(oracle)}(
      tree, oracle, workers, lock, remaining, fill_batches,
      cpuct, noise_ϵ, noise_α,
      total_time, inference_time, total_simulations, total_nodes_traversed)
  end
end

asynchronous(env::Env) = length(env.workers) > 1

GameType(::Env{Game}) where Game = Game

Done(::Env{G,B}) where {G,B} = Done{B}()
NoQuery(::Env{G,B}) where {G,B} = NoQuery{B}()

#####
##### Access and initialize board information
#####

function init_board_info(P, V)
  stats = [ActionStats(p, 0, 0, 0) for p in P]
  return BoardInfo(stats, V)
end

# Returns statistics for the current player, true if new node
# Synchronous version
function board_info_sync(env, worker, board)
  if haskey(env.tree, board)
    return (env.tree[board], false)
  else
    (P, V), time = @timed evaluate(env.oracle, board)
    env.inference_time += time
    info = init_board_info(P, V)
    env.tree[board] = info
    return (info, true)
  end
end

# Equivalent of `board_info_sync` for asynchronous MCTS
function board_info_async(env, worker, board)
  if haskey(env.tree, board)
    return (env.tree[board], false)
  else
    # Send a request to the inference server
    put!(worker.send, Query(board))
    unlock(env.global_lock)
    P, V = take!(worker.recv)
    lock(env.global_lock)
    worker.queried = true
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

function board_info(env, worker, board)
  asynchronous(env) ?
    board_info_async(env, worker, board) :
    board_info_sync(env, worker, board)
end

#####
##### Main algorithm
#####

function uct_scores(info::BoardInfo, cpuct, ϵ, η)
  @assert iszero(ϵ) || length(η) == length(info.stats)
  sqrtNtot = sqrt(Ntot(info))
  return map(enumerate(info.stats)) do (i, a)
    Q = (a.W - a.nworkers) / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    Q + cpuct * P * sqrtNtot / (a.N + 1)
  end
end

function push_board_action!(env, worker, (b, wp, aid))
  push!(worker.stack, (b, wp, aid))
  stats = env.tree[b].stats
  astats = stats[aid]
  stats[aid] = ActionStats(
    astats.P, astats.W, astats.N + 1, astats.nworkers + 1)
end

function select!(env, worker, state, η)
  state = copy(state)
  env.total_simulations += 1
  isroot = true
  while true
    wr = GI.white_reward(state)
    isnothing(wr) || (return wr)
    wp = GI.white_playing(state)
    board = GI.canonical_board(state)
    actions = GI.available_actions(state)
    let (info, new_node) = board_info(env, worker, board)
      new_node && (return info.Vest)
      ϵ = isroot ? env.noise_ϵ : 0.
      scores = uct_scores(info, env.cpuct, ϵ, η)
      best_action_id = argmax(scores)
      best_action = actions[best_action_id]
      push_board_action!(env, worker, (board, wp, best_action_id))
      GI.play!(state, best_action)
      env.total_nodes_traversed += 1
      isroot = false
    end
  end
end

function backprop!(env, worker, white_reward)
  while !isempty(worker.stack)
    board, white_playing, action_id = pop!(worker.stack)
    reward = white_playing ?
      white_reward :
      GI.symmetric_reward(white_reward)
    stats = env.tree[board].stats
    astats = stats[action_id]
    stats[action_id] = ActionStats(
      astats.P, astats.W + reward, astats.N, astats.nworkers - 1)
  end
end

# It is important to guarantee that a worker sends one request to the
# inference server per simulation. Otherwise, a worker could block all
# the others if it repeatedly doesn't need to make evaluation queries.
function worker_yield!(env::Env, worker::Worker)
  if asynchronous(env) && !worker.queried
    put!(worker.send, NoQuery(env))
    unlock(env.global_lock)
    take!(worker.recv)
    lock(env.global_lock)
  end
end

function worker_explore!(env::Env, worker::Worker, state, η)
  @assert isempty(worker.stack)
  worker.queried = false
  white_reward = select!(env, worker, state, η)
  worker_yield!(env, worker)
  backprop!(env, worker, white_reward)
  @assert isempty(worker.stack)
end

function inference_server(env::Env{G, B, A}) where {G, B, A}
  to_watch = env.workers
  while true
    # Updating the list of workers to watch
    requests = [take!(w.send) for w in to_watch]
    done = [isa(r, Done) for r in requests]
    active = .~ done
    any(active) || break
    to_watch = to_watch[active]
    requests = requests[active]
    # Gathering queries
    batch = [q.board for q in requests if isa(q, Query)]
    if isempty(batch)
      answers, time = EvaluationResult{Float32}[], 0.
    else
      if env.fill_batches
        nmissing = length(env.workers) - length(batch)
        if nmissing > 0
          append!(batch, [batch[1] for i in 1:nmissing])
        end
        @assert length(batch) == length(env.workers)
      end
      answers, time = @timed evaluate_batch(env.oracle, batch)
    end
    dummy_answer = (Float32[], 0f0)
    env.inference_time += time
    for (i, q) in enumerate(requests)
      if isa(q, Query)
        put!(to_watch[i].recv, popfirst!(answers))
      else
        put!(to_watch[i].recv, dummy_answer)
      end
    end
  end
end

function dirichlet_noise(state, α)
  actions = GI.available_actions(state)
  n = length(actions)
  return rand(Dirichlet(n, α))
end

function explore_sync!(env::Env, state, nsims)
  η = dirichlet_noise(state, env.noise_α)
  elapsed = @elapsed for i in 1:nsims
    worker_explore!(env, env.workers[1], state, η)
  end
  env.total_time += elapsed
end

function explore_async!(env::Env, state, nsims)
  env.remaining = nsims
  η = dirichlet_noise(state, env.noise_α)
  elapsed = @elapsed begin
    @sync begin
      @async @printing_errors inference_server(env)
      for w in env.workers
        @async @printing_errors begin
          lock(env.global_lock)
          while env.remaining > 0
            env.remaining -= 1
            worker_explore!(env, w, state, η)
          end
          put!(w.send, Done(env))
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

Run `nsims` MCTS simulations from `state`.
"""
function explore!(env::Env, state, nsims)
  asynchronous(env) ?
    explore_async!(env, state, nsims) :
    explore_sync!(env, state, nsims)
end

"""
    MCTS.policy(env, state; τ=1.)

Return the recommended stochastic policy on `state`,
with temperature parameter equal to `τ`. If `τ` is zero, all the weight
goes to the action with the highest visits count.

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
  if iszero(τ)
    best = argmax([a.N for a in info.stats])
    π = zeros(length(actions))
    π[best] = 1.0
    return actions, π
  else
    τinv = 1 / τ
    Ntot = sum(a.N for a in info.stats)
    π = [(a.N / Ntot) ^ τinv for a in info.stats]
    π ./= sum(π)
  end
  return actions, π
end

"""
    MCTS.reset!(env)

Empty the MCTS tree.
"""
function reset!(env)
  empty!(env.tree)
  GC.gc(true)
end

#####
##### Profiling Utilities
#####

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
MCTS simulation, not counting the root.
"""
function average_exploration_depth(env)
  return env.total_nodes_traversed / env.total_simulations
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

"""
    MCTS.approximate_memory_footprint(env)

Return an estimate of the memory footprint of the MCTS tree (in bytes).
"""
function approximate_memory_footprint(env::Env)
  return memory_footprint_per_node(env) * length(env.tree)
end

# Possibly very slow for large trees
memory_footprint(env::Env) = Base.summarysize(env.tree)

end
