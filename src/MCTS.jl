"""
    MCTS

A generic implementation of Monte Carlo Tree Search (with an external oracle).
Relies on `AlphaZero.GameInterface`.

"""
module MCTS

using DataStructures: Stack
using Distributions: Categorical

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


#####
##### MCTS Environment
#####

struct ActionStats
  P :: Float64
  W :: Float64
  N :: Int
end

mutable struct BoardInfo{Action}
  actions :: Vector{Action}
  stats   :: Vector{ActionStats}
  Ntot    :: Int
  Vest    :: Float64
end

mutable struct Env{Game, Board, Action}
  # Store state statistics assuming player one is to play for nonterminal states
  tree  :: Dict{Board, BoardInfo{Action}}
  stack :: Stack{Tuple{Board, Bool, Action}}
  cpuct :: Float64
  # External oracle to evaluate positions
  oracle :: Oracle{Game}
  function Env{G}(oracle, cpuct=1.0) where {G}
    B = GI.Board(G)
    A = GI.Action(G)
    tree = Dict{B, BoardInfo{A}}()
    stack = Stack{Tuple{B, Bool, A}}()
    new{G, B, A}(tree, stack, cpuct, oracle)
  end
end


#####
##### Initialize and update board information
#####

# white is to play
function init_board_info(oracle, board, actions)
  P, V = evaluate(oracle, board, actions)
  stats = [ActionStats(p, 0, 0) for p in P]
  BoardInfo(actions, stats, 0, V)
end

function update_board_info!(info, action, reward)
  aid = findfirst(==(action), info.actions)
  stats = info.stats[aid]
  info.Ntot += 1
  info.stats[aid] = ActionStats(stats.P, stats.W + reward, stats.N + 1)
end


#####
##### Update and access state information
#####

symmetric_reward(R) = -R

symmetric(s::ActionStats) = ActionStats(s.P, symmetric_reward(s.W), s.N)

symmetric(s::BoardInfo) = StateInfo(s.actions, map(symmetric, s.stats))

# Returns statistics for the current player, true if new node
function state_info(env, state)
  b = GI.canonical_board(state)
  if haskey(env.tree, b)
    return (env.tree[b], false)
  else
    actions = GI.available_actions(state)
    info = init_board_info(env.oracle, b, actions)
    env.tree[b] = info
    return (info, true)
  end
end


#####
##### Exploration utilities
#####

function debug_tree(env::Env{Game}; k=10) where Game
  pairs = collect(env.tree)
  k = min(k, length(pairs))
  most_visited = sort(pairs, by=(x->x.second.Ntot), rev=true)[1:k]
  for (b, info) in most_visited
    println("N: ", info.Ntot)
    GI.print_state(Game(b))
  end
end


#####
##### Main algorithm
#####

function uct_scores(info::BoardInfo, cpuct)
  sqrtNtot = sqrt(info.Ntot)
  return map(info.stats) do a
    Q = a.N > 0 ? a.W / a.N : 0.
    Q + cpuct * a.P * sqrtNtot / (a.N + 1)
  end
end

function push_state_action!(env, state, a)
  b = GI.canonical_board(state)
  wp = GI.white_playing(state)
  push!(env.stack, (b, wp, a))
end

function select!(env, state)
  state = copy(state)
  while true
    wr = GI.white_reward(state)
    isnothing(wr) || (return wr)
    let (info, new_node) = state_info(env, state)
      new_node && (return info.Vest)
      scores = uct_scores(info, env.cpuct)
      best_action = info.actions[argmax(scores)]
      push_state_action!(env, state, best_action)
      GI.play!(state, best_action)
    end
  end
end

function backprop!(env, white_reward)
  while !isempty(env.stack)
    board, white_playing, action = pop!(env.stack)
    reward = white_playing ?
      white_reward :
      symmetric_reward(white_reward)
    update_board_info!(env.tree[board], action, reward)
  end
end

function explore!(env, state, nsims=1)
  for i in 1:nsims
    @assert isempty(env.stack)
    white_reward = select!(env, state)
    backprop!(env, white_reward)
    @assert isempty(env.stack)
  end
end

# Returns (actions, distr)
function policy(env, state; τ=1.0)
  info, _ = state_info(env, state)
  τinv = 1 / τ
  D = [a.N ^ τinv for a in info.stats]
  return info.actions, D ./ sum(D)
end


#####
##### MCTS AI (for illustration purposes)
#####

struct AI <: GI.Player
  env :: Env
  timeout :: Float64
  random :: Bool
  function AI(env; timeout=3., random=false)
    new(env, timeout, random)
  end
end

function GI.select_move(ai::AI, state)
  start = time()
  while time() - start < ai.timeout
    explore!(ai.env, state, 100)
  end
  actions, distr = policy(ai.env, state)
  if ai.random
    return actions[rand(Categorical(distr))]
  else
    return actions[argmax(distr)]
  end
end


end
