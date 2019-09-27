################################################################################
# MCTS.jl
# Generic Implementation for Monte Carlo Tree Search
################################################################################

module MCTS

using DataStructures: Stack

import ..GameInterface; const GI = GameInterface

################################################################################

abstract type Oracle end

function evaluate end

################################################################################

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

mutable struct Env{State, Board, Action}
  # Store state statistics assuming player one is to play for nonterminal states
  tree  :: Dict{Board, BoardInfo{Action}}
  stack :: Stack{Action}
  state :: State
  cpuct :: Float64
  # External oracle to evaluate positions
  oracle :: Oracle
  function Env{S}(oracle, cpuct=1.0) where {S}
    B = GI.Board(S)
    A = GI.Action(S)
    new{S, B, A}(Dict{B, BoardInfo{A}}(), Stack{A}(), S(), cpuct, oracle)
  end
end

################################################################################

# white is to play
function init_board_info(oracle, board, actions)
  P, V = evaluate(oracle, board, actions)
  stats = [ActionStats(p, 0, 0) for p in P]
  BoardInfo(actions, stats, 0, V)
end

function update_board_info!(info, board, action, reward)
  aid = findfirst(==(action), info.actions)
  stats = info.stats[aid]
  info.Ntot += 1
  info.stats[aid] = ActionStats(stats.P, stats.W + reward, stats.N + 1)
end

################################################################################

symmetric_reward(R) = -R

symmetric(s::ActionStats) = ActionStats(s.P, symmetric_reward(s.W), s.N)

symmetric(s::BoardInfo) = StateInfo(s.actions, map(symmetric, s.stats))

# Returns statistics for the current player, true if new node
function state_info(env)
  if GI.white_playing(env.state)
    b = GI.board(env.state)
  else
    b = GI.board_symmetric(env.state)
  end
  if haskey(env.tree, b)
    return (env.tree[b], false)
  else
    actions = GI.available_actions(env.state)
    info = init_board_info(env.oracle, b, actions)
    env.tree[b] = info
    return (info, true)
  end
end

# The statistics has to be in the tree already
function update_state_info!(env, action, white_reward)
  if GI.white_playing(env.state)
    r = white_reward
    b = GI.board(env.state)
  else
    r = symmetric_reward(white_reward)
    b = GI.board_symmetric(env.state)
  end
  update_board_info!(env.tree[b], b, action, r)
end

################################################################################

function uct_scores(info::BoardInfo, cpuct)
  sqrtNtot = sqrt(info.Ntot)
  return map(info.stats) do a
    Q = a.N > 0 ? a.W / a.N : 0.
    Q + cpuct * a.P * sqrtNtot / (a.N + 1)
  end
end

function select!(env)
  while true
    wr = GI.white_reward(env.state)
    isnothing(wr) || (return wr)
    let (info, new_node) = state_info(env)
      new_node && (return info.Vest)
      scores = uct_scores(info, env.cpuct)
      best_action = info.actions[argmax(scores)]
      push!(env.stack, best_action)
      GI.play!(env.state, best_action)
    end
  end
end

function backprop!(env, white_reward)
  while !isempty(env.stack)
    action = pop!(env.stack)
    GI.undo!(env.state, action)
    update_state_info!(env, action, white_reward)
  end
end

################################################################################

function explore!(env, state, nsims=1)
  env.state = state
  for i in 1:nsims
    @assert isempty(env.stack)
    white_reward = select!(env)
    backprop!(env, white_reward)
    @assert isempty(env.stack)
  end
end

# Returns (actions, distr)
function policy(env; τ=1.0)
  info, _ = state_info(env)
  τinv = 1 / τ
  D = [a.N ^ τinv for a in info.stats]
  return info.actions, D ./ sum(D)
end

################################################################################

end

# Some resources
# https://web.stanford.edu/~surag/posts/alphazero.html
# https://int8.io/monte-carlo-tree-search-beginners-guide/
# https://medium.com/oracledevs/lessons-from-alpha-zero

################################################################################
