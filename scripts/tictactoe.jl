################################################################################
# Implementation of the g interface for Tic Tac Toe
################################################################################

import AlphaZero.GameInterface; const GI = GameInterface

using Gobblet.TicTacToe

################################################################################

const Game = State

GI.Board(::Type{Game}) = Board

GI.Action(::Type{Game}) = Action

GI.white_playing(g::Game) = g.curplayer == Red

GI.board(g::Game) = copy(g.board)

GI.board_symmetric(g::Game) = map!(symmetric, similar(g.board), g.board)

GI.play!(g::Game, a) = execute_action!(g, a)

GI.undo!(g::Game, a) = cancel_action!(g, a)

function GI.white_reward(g::Game) :: Union{Nothing, Float64}
  g.finished || return nothing
  isnothing(g.winner) && return 0
  g.winner == Red && return 1
  return -1
end

function GI.available_actions(g::Game)
  actions = Action[]
  sizehint!(actions, NUM_POSITIONS)
  fold_actions(g, actions) do actions, a
    push!(actions, a)
  end
  return actions
end

################################################################################

GI.board_dim(::Type{Game}) = 27

GI.num_actions(::Type{Game}) = 9

GI.action(::Type{Game}, id) = AddAction(1, id)

GI.action_id(::Type{Game}, a::Action) = a.to

const flatten = collect ∘ Iterators.flatten

function GI.vectorize_board(::Type{Game}, board)
  map(board[:,l] for l in 1:NUM_LAYERS) do layer
    map(layer) do p
      Float32[isnothing(p), p == Red, p == Blue]
    end |> flatten
  end |> flatten
end

################################################################################

function GI.action_string(::Type{Game}, a)
  TicTacToe.print_pos(a.to)
end


function read_board(::Type{Game})
  str = reduce(*, ((readline() * "   ")[1:3] for i in 1:3))
  white = ['w', 'r', 'o']
  black = ['b', 'b', 'x']
  board = TicTacToe.make_board()
  for i in 1:9
    c = nothing
    str[i] ∈ white && (c = Red)
    str[i] ∈ black && (c = Blue)
    board[i,1] = c
  end
  return board
end

# Enter a state from command line (returns `nothing` if invalid)
function GI.read_state(::Type{Game})
  b = read_board(G)
  nr = count(==(Red), b[:,1])
  nb = count(==(Blue), b[:,1])
  if nr == nb # red turn
    State(b, first_player=Red)
  elseif nr == nb + 1
    State(b, first_player=Blue)
  else
    nothing
  end
end

GI.parse_action(s::State, str) = TicTacToe.parse_action(s, str)

GI.print_state(s::State) = TicTacToe.print_board(s, with_position_names=true)

################################################################################
