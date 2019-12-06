import AlphaZero.GI

using Printf
using Crayons
using StaticArrays

const NUM_COLS = 7
const NUM_ROWS = 6
const NUM_CELLS = NUM_COLS * NUM_ROWS
const TO_CONNECT = 4

const Player = UInt8
const WHITE = 0x01
const BLACK = 0x02

other(p::Player) = 0x03 - p

const Cell = UInt8
const EMPTY = 0x00
const Board = SMatrix{NUM_COLS, NUM_ROWS, Cell, NUM_CELLS}

const INITIAL_BOARD = @SMatrix zeros(Cell, NUM_COLS, NUM_ROWS)

mutable struct Game
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  actions :: Vector{Int}
end

function Game()
  board = INITIAL_BOARD
  curplayer = WHITE
  finished = false
  winner = 0x00
  actions = collect(1:NUM_COLS)
  Game(board, curplayer, finished, winner, actions)
end

GI.Board(::Type{Game}) = Board
GI.Action(::Type{Game}) = Int

Base.copy(g::Game) =
  Game(g.board, g.curplayer, g.finished, g.winner, copy(g.actions))

#####
##### Defining game rules
#####

function first_free(board, col)
  row = 1
  while row <= NUM_ROWS && board[col, row] != EMPTY
    row += 1
  end
  return row
end

function update_available_actions!(g::Game)
  g.actions = filter(1:NUM_COLS) do col
    first_free(g.board, col) <= NUM_ROWS
  end
end

GI.available_actions(g::Game) = g.actions

valid_pos((col, row)) = 1 <= col <= NUM_COLS && 1 <= row <= NUM_ROWS

function num_connected_dir(board, player, pos, dir)
  @assert board[pos...] == player
  p = pos .+ dir
  n = 0
  while valid_pos(p) && board[p...] == player
    n += 1
    p = p .+ dir
  end
  return n
end

function num_connected_axis(board, player, pos, axis)
  @assert board[pos...] == player
  num_after = num_connected_dir(board, player, pos, axis)
  num_before = num_connected_dir(board, player, pos, (0, 0) .- axis)
  return 1 + num_before + num_after
end

function winning_pattern_at(board, player, pos)
  return any(((1, 1), (1, 0), (0, 1))) do axis
    num_connected_axis(board, player, pos, axis) >= TO_CONNECT
  end
end

# Update the game status assuming g.curplayer just played at pos=(col, row)
function update_status!(g::Game, pos)
  update_available_actions!(g)
  if winning_pattern_at(g.board, g.curplayer, pos)
    g.winner = g.curplayer
    g.finished = true
  else
    g.finished = isempty(g.actions)
  end
end

function GI.play!(g::Game, col)
  row = first_free(g.board, col)
  g.board = setindex(g.board, g.curplayer, col, row)
  update_status!(g, (col, row))
  g.curplayer = other(g.curplayer)
end


function Game(board::Board; white_playing=true)
  g = Game()
  g.board = board
  g.curplayer = white_playing ? WHITE : BLACK
  update_available_actions!(g)
  isempty(g.actions) && (g.finished = true)
  for col in 1:NUM_COLS
    top = first_free(g.board, col)
    top == 1 && continue
    row = top - 1
    c = board[col, row]
    if c != EMPTY && winning_pattern_at(board, c, (col, row))
      g.winner = c
      g.finished = true
      break
    end
  end
  return g
end

GI.board(g::Game) = g.board

cell_symmetric(c::Cell) = c == EMPTY ? EMPTY : other(c)

function GI.board_symmetric(g::Game) :: Board
  return @SMatrix [
    cell_symmetric(g.board[col, row])
    for col in 1:NUM_COLS, row in 1:NUM_ROWS]
end

GI.white_playing(g::Game) = g.curplayer == WHITE

#####
##### Reward shaping
#####

function GI.white_reward(g::Game)
  if g.finished
    g.winner == WHITE && (return  1.)
    g.winner == BLACK && (return -1.)
    return 0.
  else
    return nothing
  end
end

#####
##### ML interface
#####

GI.num_actions(::Type{Game}) = NUM_COLS

GI.action_id(::Type{Game}, col) = col

GI.action(::Type{Game}, id) = id

function GI.vectorize_board(::Type{Game}, board)
  return Float32[
    board[col, row] == c
    for col in 1:NUM_COLS,
        row in 1:NUM_ROWS,
        c in [EMPTY, WHITE, BLACK]]
end

#####
##### User interface
#####

GI.action_string(::Type{Game}, a) = string(a)

function GI.parse_action(g::Game, str)
  try
    p = parse(Int, str)
    1 <= p <= NUM_COLS ? p : nothing
  catch
    nothing
  end
end

# 1 2 3 4 5 6 7
# . . . . . . .
# . . . . . . .
# . . . . . . .
# . . o x . . .
# . o o o . . .
# o x x x . x .

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red" : "Blue"
player_mark(p)  = p == WHITE ? "o" : "x"
cell_mark(c)    = c == EMPTY ? "." : player_mark(c)
cell_color(c)   = c == EMPTY ? crayon"" : player_color(c)

function GI.print_state(g::Game; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname, " plays:", crayon"reset", "\n\n")
  # Print legend
  for col in 1:NUM_COLS
    print(GI.action_string(Game, col), " ")
  end
  print("\n")
  # Print board
  for row in NUM_ROWS:-1:1
    for col in 1:NUM_COLS
      c = g.board[col, row]
      print(cell_color(c), cell_mark(c), crayon"reset", " ")
    end
    print("\n")
  end
  botmargin && print("\n")
end

function GI.read_state(::Type{Game})
  board = Array(INITIAL_BOARD)
  try
    for col in 1:NUM_COLS
      input = readline()
      for (row, c) in enumerate(input)
        c = lowercase(c)
        if c ∈ ['o', 'w', '1']
          board[col, row] = WHITE
        elseif c ∈ ['x', 'b', '2']
          board[col, row] = BLACK
        end
      end
    end
    nw = count(==(WHITE), board)
    nb = count(==(BLACK), board)
    if nw == nb
      wp = true
    elseif nw == nb + 1
      wp = false
    else
      return nothing
    end
    return Game(Board(board), white_playing=wp)
  catch e
    return nothing
  end
end
