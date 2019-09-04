################################################################################
# SimpleMCTS.jl
# Generic Implementation for Monte Carlo Tree Search
################################################################################

"""
A fast and modular implementation of the Monte Carlo Tree Search algorithm.
The interface for `MCTS.Env{State, Board, Action}` is the following:

  + `copy_state(::State) :: State`
  + `white_playing(::State) :: Bool`
  + `terminal_reward(::State) :: Union{Nothing, Float64}`
  + `board(::State)`
  + `board_symmetric(::State)`
  + `available_actions(::State) :: Vector{Action}`
  + `play!(::State, ::Action)`
  + `undo!(::State, ::Action)`

"""
module SimpleMCTS

export Env, set_root!, explore!, most_visited_action

using DataStructures: Stack

# Generative interface
function copy_state end
function white_playing end
function terminal_reward end
function board end
function board_symmetric end
function available_actions end
function play! end
function undo! end

const DEFAULT_UCT_C     = 2 * sqrt(2)

const EXPLORE_INCREMENT = 100

################################################################################

struct StateStatistics
  Q :: Float64
  N :: Int
end

mutable struct Env{State, Board, Action}
  # Store state statistics assuming player one is to play
  tree  :: Dict{Board, StateStatistics}
  stack :: Stack{Action}
  state :: State
  # Parameters
  uct_c :: Float64
  
  function Env{S, B, A}() where {S, B, A}
    new(Dict{B, StateStatistics}(), Stack{A}(), S(), DEFAULT_UCT_C)
  end
end

################################################################################

symmetric_reward(R) = -R

symmetric(s::StateStatistics) = StateStatistics(symmetric_reward(s.Q), s.N)

const INITIAL_STATE_STATISTICS = StateStatistics(0, 0)

# Returns statistics for the current player
function state_statistics(env::Env)
  b = white_playing(env.state) ? board(env.state) : board_symmetric(env.state)
  if haskey(env.tree, b)
    return env.tree[b]
  else
    return env.tree[b] = INITIAL_STATE_STATISTICS
  end
end

function set_state_statistics!(env::Env, reward)
  if white_playing(env.state)
    r = reward
    b = board(env.state)
  else
    r = symmetric_reward(reward)
    b = board_symmetric(env.state)
  end
  stats = get(env.tree, b, INITIAL_STATE_STATISTICS)
  env.tree[b] = StateStatistics(stats.Q + r, stats.N + 1)
end

################################################################################

function uct_score(cur::StateStatistics, child::StateStatistics, c)
  child.N == 0 && return Inf
  return child.Q / child.N + c * sqrt(log(cur.N) / child.N)
end

function uct_score(env::Env, curstats::StateStatistics, action)
  player = white_playing(env.state)
  play!(env.state, action)
  cstats = state_statistics(env)
  if white_playing(env.state) != player
    cstats = symmetric(cstats)
  end
  undo!(env.state, action)
  uct_score(curstats, cstats, env.uct_c)
end

function select!(env::Env)
  reward = nothing
  while true
    reward = terminal_reward(env.state)
    isnothing(reward) || break
    let stats = state_statistics(env)
      stats.N == 0 && break
      actions = available_actions(env.state)
      scores = [uct_score(env, stats, a) for a in actions]
      best_action = actions[argmax(scores)]
      push!(env.stack, best_action)
      play!(env.state, best_action)
    end
  end
  return reward
end

function rollout(env::Env)
  state = copy_state(env.state)
  while true
    reward = terminal_reward(state)
    isnothing(reward) || (return reward)
    action = rand(available_actions(state))
    play!(state, action)
  end
end

function backprop!(env::Env, reward)
  set_state_statistics!(env, reward)
  while !isempty(env.stack)
    action = pop!(env.stack)
    undo!(env.state, action)
    set_state_statistics!(env, reward)
  end
end

function explore!(env::Env)
  @assert isempty(env.stack)
  reward = select!(env)
  if isnothing(reward)
    reward = rollout(env)
  end
  backprop!(env, reward)
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

function most_visited_action(env::Env)
  actions = available_actions(env.state)
  scores = map(actions) do a
    play!(env.state, a)
    stats = state_statistics(env)
    undo!(env.state, a)
    stats.N
  end
  return actions[argmax(scores)]
end

################################################################################

end

# Some resources
# https://web.stanford.edu/~surag/posts/alphazero.html
# https://int8.io/monte-carlo-tree-search-beginners-guide/
# https://medium.com/oracledevs/lessons-from-alpha-zero

################################################################################
