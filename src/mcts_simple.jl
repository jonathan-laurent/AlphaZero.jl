"""
A generic, standalone implementation of Monte Carlo Tree Search.
It can be used on any game that implements `GameInterface`
and with any external oracle.
"""
module MCTS

using Distributions: Categorical, Dirichlet

using ..Util: apply_temperature
import ..GI, ..GameType

#####
##### Interface for External Oracles
#####

"""
    MCTS.Oracle{Game}

Abstract base type for an oracle. Oracles must be callable:

  (::Oracle)(state)

Evaluate a single state from the current player's perspective.

Return a pair `(P, V)` where:

  - `P` is a probability vector on `GI.available_actions(Game(state))`
  - `V` is a scalar estimating the value or win probability for white.
"""
abstract type Oracle{Game} end

GameType(::Oracle{G}) where G = G

"""
    MCTS.RolloutOracle{Game}(γ=1.)

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

function (r::RolloutOracle{Game})(state) where Game
  game = Game(state)
  wp = GI.white_playing(game)
  n = length(GI.available_actions(game))
  P = ones(n) ./ n
  wr = rollout!(game, r.gamma)
  V = wp ? wr : -wr
  return P, V
end

struct RandomOracle{Game} <: Oracle{Game} end

function (::RandomOracle{Game})(state) where Game
  s = Game(state)
  n = length(GI.available_actions(s))
  P = ones(n) ./ n
  V = 0.
  return P, V
end

#####
##### State Statistics
#####

struct ActionStats
  P :: Float32 # Prior probability as given by the oracle
  W :: Float64 # Cumulated Q-value for the action (Q = W/N)
  N :: Int # Number of times the action has been visited
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

  - `gamma=1.`: the reward discount factor
  - `cpuct=1.`: exploration constant in the UCT formula
  - `noise_ϵ=0., noise_α=1.`: parameters for the dirichlet exploration noise
     (see below)
  - `prior_temperature=1.`: temperature to apply to the oracle's output
     to get the prior probability vector used by MCTS.

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
  # Parameters
  gamma :: Float64 # Discount factor
  cpuct :: Float64
  noise_ϵ :: Float64
  noise_α :: Float64
  prior_temperature :: Float64
  # Performance statistics
  total_simulations :: Int64
  total_nodes_traversed :: Int64

  function Env{G}(oracle;
      gamma=1., cpuct=1., noise_ϵ=0., noise_α=1., prior_temperature=1.) where G
    S = GI.State(G)
    tree = Dict{S, StateInfo}()
    total_simulations = 0
    total_nodes_traversed = 0
    new{G, S, typeof(oracle)}(
      tree, oracle, gamma, cpuct, noise_ϵ, noise_α, prior_temperature,
      total_simulations, total_nodes_traversed)
  end
end

GameType(::Env{Game}) where Game = Game

#####
##### Access and initialize state information
#####

function init_state_info(P, V, prior_temperature)
  P = apply_temperature(P, prior_temperature)
  stats = [ActionStats(p, 0, 0) for p in P]
  return StateInfo(stats, V)
end

# Returns statistics for the current player, along with a boolean indicating
# whether or not a new node has been created.
# Synchronous version
function state_info(env, state)
  if haskey(env.tree, state)
    return (env.tree[state], false)
  else
    (P, V) = env.oracle(state)
    info = init_state_info(P, V, env.prior_temperature)
    env.tree[state] = info
    return (info, true)
  end
end

#####
##### Main algorithm
#####

function uct_scores(info::StateInfo, cpuct, ϵ, η)
  @assert iszero(ϵ) || length(η) == length(info.stats)
  sqrtNtot = sqrt(Ntot(info))
  return map(enumerate(info.stats)) do (i, a)
    Q = a.W / max(a.N, 1)
    P = iszero(ϵ) ? a.P : (1-ϵ) * a.P + ϵ * η[i]
    Q + cpuct * P * sqrtNtot / (a.N + 1)
  end
end

# Also decreases the visit count
function update_state_info!(env, state, action_id, q)
  stats = env.tree[state].stats
  astats = stats[action_id]
  stats[action_id] = ActionStats(astats.P, astats.W + q, astats.N + 1)
end

# Run a single MCTS simulation, updating the statistics of all traversed states.
# Return the estimated Q-value for the current player.
# Modifies the state of the game.
function run_simulation!(env::Env, game; η, root=true)
  if GI.game_terminated(game)
    return 0.
  else
    state = GI.current_state(game)
    actions = GI.available_actions(game)
    info, new_node = state_info(env, state)
    if new_node
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
      qnext = run_simulation!(env, game, η=η, root=false)
      qnext = pswitch ? -qnext : qnext
      q = r + env.gamma * qnext
      update_state_info!(env, state, action_id, q)
      env.total_nodes_traversed += 1
      return q
    end
  end
end

function dirichlet_noise(game, α)
  actions = GI.available_actions(game)
  n = length(actions)
  return rand(Dirichlet(n, α))
end

"""
    MCTS.explore!(env, game, nsims)

Run `nsims` MCTS simulations from the current state.
"""
function explore!(env::Env, game, nsims)
  η = dirichlet_noise(game, env.noise_α)
  for i in 1:nsims
    env.total_simulations += 1
    run_simulation!(env, copy(game), η=η)
  end
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
  #GC.gc(true)
end

#####
##### Profiling Utilities
#####

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
    ActionStats(0, 0, 0) for i in 1:GI.num_actions(G)], 0)
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
