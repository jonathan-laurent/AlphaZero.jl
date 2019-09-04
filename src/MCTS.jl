################################################################################
# MCTS.jl
# Generic Implementation for Monte Carlo Tree Search
################################################################################

"""
A fast and modular implementation of the Monte Carlo Tree Search algorithm for
symmetric zero-sum games.
The interface for `MCTS.Env{State, Board, Action, Evaluator}` is the following:

  + `white_playing(::State) :: Bool`
  + `white_reward(::State) :: Union{Nothing, Float64}`
  + `board(::State)`
  + `board_symmetric(::State)`
  + `available_actions(::State) :: Vector{Action}`
  + `play!(::State, ::Action)`
  + `undo!(::State, ::Action)`
  + `evaluate(::Evaluator, ::Board, available_actions) -> (P, V)`

Actions must be symmetric in the sense that they do not depend on the current
player (they are expressed in relative terms).
We expect the following to hold:
  available_actions(s) =
    available_actions(State(board_symmetric(s), player=symmetric(s.curplayer)))
"""
module MCTS

export Env, set_root!, explore!, policy

using DataStructures: Stack

# Generative interface
function copy_state end
function white_playing end
function white_reward end
function board end
function board_symmetric end
function available_actions end
function play! end
function undo! end
function evaluate end

const CPUCT = 1.0
const EXPLORE_INCREMENT = 1000

################################################################################

struct ActionStatistics
  P :: Float64
  W :: Float64
  N :: Int
end

mutable struct BoardInfo{Action}
  actions :: Vector{Action}
  stats   :: Vector{ActionStatistics}
  Ntot    :: Int
  Vest    :: Float64
end

mutable struct Env{State, Board, Action, Evaluator}
  # Store state statistics assuming player one is to play for nonterminal states
  tree  :: Dict{Board, BoardInfo{Action}}
  stack :: Stack{Action}
  state :: State
  evaluator :: Evaluator
  function Env{S, B, A, E}(evaluator::E) where {S, B, A, E}
    new(Dict{B, BoardInfo{A}}(), Stack{A}(), S(), evaluator)
  end
end

################################################################################

# white is to play
function init_board_info(evaluator, board, actions)
  P, V = evaluate(evaluator, board, actions)
  stats = [ActionStatistics(p, 0, 0) for p in P]
  BoardInfo(actions, stats, 0, V)
end

function update_board_info!(info, board, action, reward)
  aid = findfirst(==(action), info.actions)
  stats = info.stats[aid]
  info.Ntot += 1
  info.stats[aid] = ActionStatistics(stats.P, stats.W + reward, stats.N + 1)
end

################################################################################

symmetric_reward(R) = -R

symmetric(s::ActionStatistics) =
  ActionStatistics(s.P, symmetric_reward(s.W), s.N)

symmetric(s::BoardInfo) = StateInfo(s.actions, map(symmetric, s.stats))

# Returns statistics for the current player, true if new node
function state_info(env::Env)
  if white_playing(env.state)
    b = board(env.state)
  else
    b = board_symmetric(env.state)
  end
  if haskey(env.tree, b)
    return (env.tree[b], false)
  else
    actions = available_actions(env.state)
    info = init_board_info(env.evaluator, b, actions)
    env.tree[b] = info
    return (info, true)
  end
end

# The statistics has to be in the tree already
function update_state_info!(env::Env, action, white_reward)
  if white_playing(env.state)
    r = white_reward
    b = board(env.state)
  else
    r = symmetric_reward(white_reward)
    b = board_symmetric(env.state)
  end
  update_board_info!(env.tree[b], b, action, r)
end

################################################################################

function uct_scores(info::BoardInfo)
  sqrtNtot = sqrt(info.Ntot)
  return map(info.stats) do a
    Q = a.N > 0 ? a.W / a.N : 0.
    Q + CPUCT * a.P * sqrtNtot / (a.N + 1)
  end
end

function select!(env::Env)
  while true
    wr = white_reward(env.state)
    isnothing(wr) || (return wr)
    let (info, new_node) = state_info(env)
      new_node && (return info.Vest)
      scores = uct_scores(info)
      best_action = info.actions[argmax(scores)]
      push!(env.stack, best_action)
      play!(env.state, best_action)
    end
  end
end

function backprop!(env::Env, white_reward)
  while !isempty(env.stack)
    action = pop!(env.stack)
    undo!(env.state, action)
    update_state_info!(env, action, white_reward)
  end
end

function explore!(env::Env)
  @assert isempty(env.stack)
  white_reward = select!(env)
  backprop!(env, white_reward)
  @assert isempty(env.stack)
end

################################################################################

function explore!(env::Env, time_budget)
  start = time()
  while time() - start <= time_budget
    for i in 1:EXPLORE_INCREMENT
      explore!(env)
    end
  end
end

function set_root!(env::Env, state)
  env.state = state
end

# Returns (actions, distr)
function policy(env::Env, τ=1.0)
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
