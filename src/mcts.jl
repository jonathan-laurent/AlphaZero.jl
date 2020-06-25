"""
A generic, standalone implementation of Monte Carlo Tree Search.
It can be used on any game that implements `GameInterface`
and with any external oracle.

Both a synchronous and an asynchronous version are implemented, which
share most of their code. When browsing the sources for the first time,
we recommend that you study the sychronous version first.
"""
module MCTS

using DataStructures: Stack
using Distributions: Categorical, Dirichlet

using ..Util: @printing_errors, apply_temperature
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
    MCTS.evaluate(oracle::Oracle, state)

Evaluate a single state from the current player's perspective.

Return a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(Game(state))`
  - `V` is a scalar estimating the value or win probability for white.
"""
function evaluate end

"""
    MCTS.evaluate_batch(oracle::Oracle, states)

Evaluate a batch of states.

Expect a vector of states and return a vector of `(P, V)` pairs.

A default implementation is provided that calls [`MCTS.evaluate`](@ref)
sequentially on each position.
"""
function evaluate_batch(oracle::Oracle, states)
  return [evaluate(oracle, b) for b in states]
end

GameType(::Oracle{Game}) where Game = Game

#####
##### Some Example Oracles
#####

"""
    MCTS.RolloutOracle{Game}(γ=1.) <: MCTS.Oracle{Game}

This oracle estimates the value of a position by simulating a random game
from it (a rollout). Moreover, it puts a uniform prior on available actions.
Therefore, it can be used to implement the "vanilla" MCTS algorithm.
"""
struct RolloutOracle{Game} <: Oracle{Game}
  gamma :: Float64
  RolloutOracle{G}(γ=1.) where G = new{G}(γ)
end

function rollout!(game, γ=1.)
  r = 0.
  while !GI.game_terminated(game)
    action = rand(GI.available_actions(game))
    GI.play!(game, action)
    r = γ * r + GI.white_reward(game)
  end
  return r
end

function evaluate(r::RolloutOracle{Game}, state) where Game
  game = Game(state)
  wp = GI.white_playing(game)
  n = length(GI.available_actions(game))
  P = ones(n) ./ n
  wr = rollout!(game, r.gamma)
  V = wp ? wr : -wr
  return P, V
end

struct RandomOracle{Game} <: Oracle{Game} end

function evaluate(::RandomOracle{Game}, state) where Game
  s = Game(state)
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

abstract type Message{S} end
struct Done{S} <: Message{S} end
struct NoQuery{S} <: Message{S} end
struct Query{S} <: Message{S} state :: S end
const AnyMessage{S} = Union{Done{S}, NoQuery{S}, Query{S}}
const EvaluationResult{R} = Tuple{Vector{R}, R}

mutable struct Worker{State}
  id :: Int # useful for debugging purposes
  send :: Channel{AnyMessage{State}}
  recv :: Channel{EvaluationResult{Float32}}
  queried :: Bool # the worker queried the server during the current simulation

  function Worker{S}(id) where S
    send = Channel{AnyMessage{S}}(1)
    recv = Channel{EvaluationResult{Float32}}(1)
    new{S}(id, send, recv, false)
  end
end

#####
##### State Statistics
#####

struct ActionStats
  P :: Float32 # Prior probability as given by the oracle
  W :: Float64 # Cumulated Q-value for the action (Q = W/N)
  N :: Int # Number of times the action has been visited
  nworkers :: UInt16 # Number of workers currently exploring this branch
end

struct StateInfo
  stats :: Vector{ActionStats}
  Vest  :: Float32 # Value estimate given by the oracle
end

Ntot(b::StateInfo) = sum(s.N for s in b.stats)

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
  - `gamma=1.`: the reward discount factor
  - `cpuct=1.`: exploration constant in the UCT formula
  - `noise_ϵ=0., noise_α=1.`: parameters for the dirichlet exploration noise
     (see below)
  - `prior_temperature=1.`: temperature to apply to the oracle's output
     to get the prior probability vector used by MCTS.

## Asynchronous MCTS

  - If `nworkers == 1`, MCTS is run in a synchronous fashion and the oracle is
    invoked through [`MCTS.evaluate`](@ref).

  - If `nworkers > 1`, `nworkers` asynchronous workers are spawned,
    along with an additional task to serve state evaluation requests.
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
mutable struct Env{Game, State, Oracle}
  # Store (nonterminal) state statistics assuming player one is to play
  tree :: Dict{State, StateInfo}
  # External oracle to evaluate positions
  oracle :: Oracle
  # Workers
  workers :: Vector{Worker{State}}
  global_lock :: ReentrantLock # This impl. is asynchronous but sequential
  remaining :: Int # counts the number of remaining simulations to do
  # Parameters
  fill_batches :: Bool
  gamma :: Float64 # Discount factor
  cpuct :: Float64
  noise_ϵ :: Float64
  noise_α :: Float64
  prior_temperature :: Float64
  # Performance statistics
  total_time :: Float64
  inference_time :: Float64
  total_simulations :: Int64
  total_nodes_traversed :: Int64

  function Env{G}(oracle;
      nworkers=1, fill_batches=false, gamma=1.,
      cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.) where G
    S = GI.State(G)
    tree = Dict{S, StateInfo}()
    total_time = 0.
    inference_time = 0.
    total_simulations = 0
    total_nodes_traversed = 0
    lock = ReentrantLock()
    remaining = 0
    workers = [Worker{S}(i) for i in 1:nworkers]
    new{G, S, typeof(oracle)}(
      tree, oracle, workers, lock, remaining, fill_batches, gamma,
      cpuct, noise_ϵ, noise_α, prior_temperature,
      total_time, inference_time, total_simulations, total_nodes_traversed)
  end
end

asynchronous(env::Env) = length(env.workers) > 1

GameType(::Env{Game}) where Game = Game

Done(::Env{G,S}) where {G,S} = Done{S}()
NoQuery(::Env{G,S}) where {G,S} = NoQuery{S}()

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature)
  P = apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0, 0) for p in P]
  return StateInfo(stats, V)
end

# Returns statistics for the current player, along with a boolean indicating
# whether or not a new node has been created.
# Synchronous version
function state_info_sync(env, worker, state)
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    (P, V), time = @timed evaluate(env.oracle, state)
    env.inference_time += time
    info = init_state_info(P, V, env.prior_temperature)
    env.tree[state] = info
    return (info, true)
  end
end

# Equivalent of `state_info_sync` for asynchronous MCTS
function state_info_async(env, worker, state)
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    # Send a request to the inference server
    put!(worker.send, Query(state))
    unlock(env.global_lock)
    P, V = take!(worker.recv)
    lock(env.global_lock)
    worker.queried = true
    # Another worker may have sent the same request and initialized
    # the node before. Therefore, we have to test membership again.
    if !haskey(env.tree, state)
      info = init_state_info(P, V, env.prior_temperature)
      env.tree[state] = info
      return (info, true)
    else
      # The inference result is ignored and we proceed as if
      # the node was already in the tree.
      return (env.tree[state], false)
    end
  end
end

function state_info(env, worker, state)
  asynchronous(env) ?
    state_info_async(env, worker, state) :
    state_info_sync(env, worker, state)
end

#####
##### Main algorithm
#####

function uct_scores(info::StateInfo, cpuct, ϵ, η)
  @assert iszero(ϵ) || length(η) == length(info.stats)
  sqrtNtot = sqrt(Ntot(info))
  return map(enumerate(info.stats)) do (i, a)
    Q = (a.W - a.nworkers) / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    Q + cpuct * P * sqrtNtot / (a.N + 1)
  end
end

function current_player_reward(game)
  wr = GI.white_reward(game)
  return GI.white_playing(game) ? wr : -wr
end

function increment_visit_counter!(env, state, action_id)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(
    astats.P, astats.W, astats.N + 1, astats.nworkers + 1)
end

# Also decreases the visit count
function update_state_info!(env, state, action_id, q)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(
    astats.P, astats.W + q, astats.N, astats.nworkers - 1)
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

# Run a single MCTS simulation, updating the statistics of all traversed states.
# Return the estimated Q-value for the current player.
# Leave the game unchanged (a copy is made while visiting the root).
function worker_explore!(env::Env, worker::Worker, game, η, root=true)
  if root
    worker.queried = false
    env.total_simulations += 1
    game = copy(game)
  end
  if GI.game_terminated(game)
    worker_yield!(env, worker)
    return 0.
  else
    state = GI.current_state(game)
    actions = GI.available_actions(game)
    info, new_node = state_info(env, worker, state)
    if new_node
      worker_yield!(env, worker)
      return info.Vest
    else
      ϵ = root ? env.noise_ϵ : 0.
      scores = uct_scores(info, env.cpuct, ϵ, η)
      action_id = argmax(scores)
      action = actions[action_id]
      wp = GI.white_playing(game)
      GI.play!(game, action)
      wr = GI.white_reward(game)
      r = wp ? wr : -wr
      pswitch = wp != GI.white_playing(game)
      increment_visit_counter!(env, state, action_id)
      qnext = worker_explore!(env, worker, game, η, false)
      qnext = pswitch ? -qnext : qnext
      q = r + env.gamma * qnext
      update_state_info!(env, state, action_id, q)
      env.total_nodes_traversed += 1
      return q
    end
  end
end

function inference_server(env::Env{G, S, A}) where {G, S, A}
  to_watch = env.workers
  while true
    # Update the list of workers to watch (the ones that are not done)
    requests = [take!(w.send) for w in to_watch]
    done = [isa(r, Done) for r in requests]
    active = .~ done
    any(active) || break
    to_watch = to_watch[active]
    requests = requests[active]
    # Gather a batch of queries and evaluate them
    batch = [q.state for q in requests if isa(q, Query)]
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
    env.inference_time += time
    # Send responses to all workers. Workers that sent `NoQuery` receive
    # a dummy answer, which they ignore.
    dummy_answer = (Float32[], 0f0)
    for (i, q) in enumerate(requests)
      if isa(q, Query)
        put!(to_watch[i].recv, popfirst!(answers))
      else
        put!(to_watch[i].recv, dummy_answer)
      end
    end
  end
end

function dirichlet_noise(game, α)
  actions = GI.available_actions(game)
  n = length(actions)
  return rand(Dirichlet(n, α))
end

function explore_sync!(env::Env, game, nsims)
  η = dirichlet_noise(game, env.noise_α)
  elapsed = @elapsed for i in 1:nsims
    worker_explore!(env, env.workers[1], game, η)
  end
  env.total_time += elapsed
end

# Spawn an inference server along with `env.nworkers` workers.
# Each worker runs simulations in a loop until `env.remaining` reaches zero.
function explore_async!(env::Env, game, nsims)
  env.remaining = nsims
  η = dirichlet_noise(game, env.noise_α)
  elapsed = @elapsed begin
    @sync begin
      @async @printing_errors inference_server(env)
      for w in env.workers
        @async @printing_errors begin
          lock(env.global_lock)
          while env.remaining > 0
            env.remaining -= 1
            worker_explore!(env, w, game, η)
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
    MCTS.explore!(env, game, nsims)

Run `nsims` MCTS simulations from the current state.
"""
function explore!(env::Env, game, nsims)
  asynchronous(env) ?
    explore_async!(env, game, nsims) :
    explore_sync!(env, game, nsims)
end

"""
    MCTS.policy(env, game)

Return the recommended stochastic policy on the current state.

A call to this function must always be preceded by
a call to [`MCTS.explore!`](@ref).
"""
function policy(env::Env, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  info =
    try env.tree[state]
    catch e
      if isa(e, KeyError)
        error("MCTS.explore! must be called before MCTS.policy")
      else
        rethrow(e)
      end
    end
  Ntot = sum(a.N for a in info.stats)
  π = [a.N / Ntot for a in info.stats]
  π ./= sum(π)
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
  # For every element, a state and a pointer are stored
  size_key = 2 * (GI.state_memsize(G) + sizeof(Int))
  dummy_stats = StateInfo([
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
