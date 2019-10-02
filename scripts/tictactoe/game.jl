################################################################################
# Simple Tic Tac Toe Implementation
################################################################################

module TicTacToe

import AlphaZero.GI

using StaticArrays

const BOARD_SIDE = 3

const NUM_POSITIONS = BOARD_SIDE ^ 2

const Cell = Union{Nothing, Bool}

const Board = SVector{NUM_POSITIONS, Cell}

const Player = Bool

const WHITE = true

const BLACK = false

const INITIAL_BOARD = Board(repeat([nothing], NUM_POSITIONS))

mutable struct Game
  board     :: Board
  curplayer :: Player
  finished  :: Bool
  winner    :: Union{Nothing, Bool}
  free      :: Vector{Int}
  function Game()
    new(INITIAL_BOARD, WHITE, false, nothing, collect(1:NUM_POSITIONS))
  end
end

GI.Board(::Type{Game}) = Board

GI.Action(::Type{Game}) = Int

################################################################################

pos_of_xy((x, y)) = (y - 1) * BOARD_SIDE + (x - 1) + 1

xy_of_pos(pos) = ((pos - 1) % BOARD_SIDE + 1, (pos - 1) ÷ BOARD_SIDE + 1)

const ALIGNMENTS =
  let N = BOARD_SIDE
  let XY = [
    [[(i, j) for j in 1:N] for i in 1:N];
    [[(i, j) for i in 1:N] for j in 1:N];
    [[(i, i) for i in 1:N]];
    [[(i, N - i + 1) for i in 1:N]]]
  [map(pos_of_xy, al) for al in XY]
end end

function has_won(g::Game, player)
  any(ALIGNMENTS) do al
    all(al) do pos
      g.board[pos] == player
    end
  end
end

function update_free!(g::Game)
  g.free = findall(==(nothing), g.board)
end

################################################################################

function Game(board, player=WHITE)
  g = Game()
  g.board = board
  update_free!(g)
  if isempty(g.free)
    g.finished = true
    g.winner = nothing
  elseif has_won(g, WHITE)
    g.finished = true
    g.winner = WHITE
  elseif has_won(g, BLACK)
    g.finished = true
    g.winner = BLACK
  end
  return g
end

GI.available_actions(g::Game) = g.free

GI.board(g::Game) = g.board

function GI.board_symmetric(g::Game)
  symmetric(c::Cell) = isnothing(c) ? nothing : !c
  @SVector Cell[symmetric(g.board[i]) for i in 1:NUM_POSITIONS]
end

GI.white_playing(g::Game) = g.curplayer

function GI.white_reward(g::Game)
  g.finished || return nothing
  isnothing(g.winner) && return 0.
  return g.winner ? 1. : -1.
end

function GI.play!(g::Game, pos)
  g.board = setindex(g.board, g.curplayer, pos)
  update_free!(g)
  if isempty(g.free)
    g.finished = true
    g.winner = nothing
  elseif has_won(g, g.curplayer)
    g.finished = true
    g.winner = g.curplayer
  end
  g.curplayer = !g.curplayer
end

function GI.undo!(g::Game, pos)
  g.curplayer = !g.curplayer
  g.finished = false
  g.winner = nothing
  g.board = setindex(g.board, nothing, pos)
  update_free!(g)
end

################################################################################

GI.board_dim(::Type{Game}) = 3 * NUM_POSITIONS

GI.num_actions(::Type{Game}) = NUM_POSITIONS

GI.action(::Type{Game}, id) = id

GI.action_id(::Type{Game}, pos) = pos

function GI.vectorize_board(::Type{Game}, ::Type{R}, board) where R
  R[ board[i] == c
     for c in [nothing, WHITE, BLACK]
     for i in 1:NUM_POSITIONS ]
end

################################################################################

function GI.action_string(::Type{Game}, a)
  Char(Int('A') + a - 1)
end

function GI.parse_action(g::Game, str)
  length(str) == 1 || (return nothing)
  x = Int(uppercase(str[1])) - Int('A')
  (0 <= x < NUM_POSITIONS) ? x + 1 : nothing
end

function read_board(::Type{Game})
  n = BOARD_SIDE
  str = reduce(*, ((readline() * "   ")[1:n] for i in 1:n))
  white = ['w', 'r', 'o']
  black = ['b', 'b', 'x']
  function cell(i)
    if (str[i] ∈ white) WHITE
    elseif (str[i] ∈ black) BLACK
    else nothing end
  end
  @SVector [cell(i) for i in 1:NUM_POSITIONS]
end

function GI.read_state(::Type{Game})
  b = read_board(Game)
  nw = count(==(WHITE), b)
  nb = count(==(BLACK), b)
  if nw == nb
    Game(b, WHITE)
  elseif nw == nb + 1
    Game(b, BLACK)
  else
    nothing
  end
end

using Crayons

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red" : "Blue"
player_mark(p)  = p == WHITE ? "o" : "x"

function GI.print_state(g::Game; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname, " plays:", crayon"reset", "\n\n")
  for y in 1:BOARD_SIDE
    for x in 1:BOARD_SIDE
      pos = pos_of_xy((x, y))
      c = g.board[pos]
      if isnothing(c)
        print(" ")
      else
        print(player_color(c), player_mark(c), crayon"reset")
      end
      print(" ")
    end
    if with_position_names
      print(" | ")
      for x in 1:BOARD_SIDE
        print(GI.action_string(Game, pos_of_xy((x, y))), " ")
      end
    end
    print("\n")
  end
  botmargin && print("\n")
end

end

################################################################################
