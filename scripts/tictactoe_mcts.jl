################################################################################
# Solving simple tic tac toe with MCTS
################################################################################

using AlphaZero.SimpleMCTS

using Gobblet.TicTacToe

################################################################################

const PROFILE = false
const INTERACTIVE = true

const INITIAL_TRAINING = 1. # second
const TIMEOUT = 0.5 # seconds

################################################################################
  
SimpleMCTS.copy_state(s::State) = deepcopy(s)

SimpleMCTS.white_playing(s::State) = s.curplayer == Red

SimpleMCTS.board(s::State) = copy(s.board)

SimpleMCTS.board_symmetric(s::State) = map!(symmetric, similar(s.board), s.board)

SimpleMCTS.play!(s::State, a) = execute_action!(s, a)

SimpleMCTS.undo!(s::State, a) = cancel_action!(s, a)

function SimpleMCTS.terminal_reward(s::State) :: Union{Nothing, Float64}
  s.finished || return nothing
  isnothing(s.winner) && return 0
  s.winner == Red && return 1
  return -1
end

function SimpleMCTS.available_actions(s::State)
  actions = Action[]
  sizehint!(actions, NUM_POSITIONS)
  fold_actions(s, actions) do actions, a
    push!(actions, a)
  end
  return actions
end

################################################################################

const GobbletMCTS = SimpleMCTS.Env{State, Board, Action}

struct MonteCarloAI <: AI
  env :: GobbletMCTS
  timeout :: Float64
end

import Gobblet.TicTacToe: play

function play(ai::MonteCarloAI, state)
  SimpleMCTS.set_root!(ai.env, state)
  SimpleMCTS.explore!(ai.env, ai.timeout)
  SimpleMCTS.most_visited_action(ai.env)
end

################################################################################

function debug_tree(env, k=10)
  pairs = collect(env.tree)
  k = min(k, length(pairs))
  most_visited = sort(pairs, by=(x->x.second.N), rev=true)[1:k]
  for (b, stats) in most_visited
    println(stats)
    print_board(State(b))
  end
end

################################################################################

# In our experiments, we can simulate ~10000 games per second

using Profile
using ProfileView

if PROFILE
  env = GobbletMCTS()
  SimpleMCTS.explore(env, 0.1)
  Profile.clear()
  @profile SimpleMCTS.explore(env, 0.5)
  ProfileView.svgwrite("profile_mcts.svg")
  # To examine code:
  # code_warntype(SimpleMCTS.select!, Tuple{GobbletMCTS})
end

env = GobbletMCTS()
state = State()
SimpleMCTS.set_root!(env, state)
SimpleMCTS.explore!(env, INITIAL_TRAINING)

if INTERACTIVE
  interactive!(state, red=MonteCarloAI(env, TIMEOUT), blue=Human())
end

################################################################################
