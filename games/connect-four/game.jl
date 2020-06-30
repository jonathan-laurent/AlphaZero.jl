import AlphaZero.GI

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
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

mutable struct Game <: GI.AbstractGame
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  amask :: Vector{Bool} # actions mask
  # Actions history, which uniquely identifies the current board position
  # Used by external solvers
  history :: Union{Nothing, Vector{Int}}
end

function Game()
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = 0x00
  amask = trues(NUM_COLS)
  history = Int[]
  Game(board, curplayer, finished, winner, amask, history)
end

GI.State(::Type{Game}) = typeof(INITIAL_STATE)

GI.Action(::Type{Game}) = Int

GI.two_players(::Type{Game}) = true

const ACTIONS = collect(1:NUM_COLS)

GI.actions(::Type{Game}) = ACTIONS

function Base.copy(g::Game)
  history = isnothing(g.history) ? nothing : copy(g.history)
  Game(g.board, g.curplayer, g.finished, g.winner, copy(g.amask), history)
end

history(g::Game) = g.history

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

function update_actions_mask!(g::Game)
  g.amask = map(ACTIONS) do col
    first_free(g.board, col) <= NUM_ROWS
  end
end

GI.actions_mask(g::Game) = g.amask

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
  return any(((1, 1), (1, -1), (1, 0), (0, 1))) do axis
    num_connected_axis(board, player, pos, axis) >= TO_CONNECT
  end
end

# Update the game status assuming g.curplayer just played at pos=(col, row)
function update_status!(g::Game, pos)
  update_actions_mask!(g)
  if winning_pattern_at(g.board, g.curplayer, pos)
    g.winner = g.curplayer
    g.finished = true
  else
    g.finished = !any(g.amask)
  end
end

function GI.play!(g::Game, col)
  isnothing(g.history) || push!(g.history, col)
  row = first_free(g.board, col)
  g.board = setindex(g.board, g.curplayer, col, row)
  update_status!(g, (col, row))
  g.curplayer = other(g.curplayer)
end

function Game(state)
  board = state.board
  g = Game()
  g.history = nothing
  g.board = board
  g.curplayer = state.curplayer
  update_actions_mask!(g)
  any(g.amask) || (g.finished = true)
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

GI.current_state(g::Game) = (board=g.board, curplayer=g.curplayer)

GI.white_playing(::Type{Game}, state) = state.curplayer == WHITE

#####
##### Reward shaping
#####

function GI.game_terminated(g::Game)
  return g.finished
end

function GI.white_reward(g::Game)
  if g.finished
    g.winner == WHITE && (return  1.)
    g.winner == BLACK && (return -1.)
    return 0.
  else
    return 0.
  end
end

#####
##### Simple heuristic for minmax
#####

const Pos = Tuple{Int, Int}
const Alignment = Vector{Pos}

function alignment_from(pos, dir) :: Union{Alignment, Nothing}
  al = Alignment()
  for i in 1:TO_CONNECT
    valid_pos(pos) || (return nothing)
    push!(al, pos)
    pos = pos .+ dir
  end
  return al
end

function alignments_with(dir) :: Vector{Alignment}
  als = [alignment_from((x, y), dir) for x in 1:NUM_COLS for y in 1:NUM_ROWS]
  return filter(al -> !isnothing(al), als)
end

const ALIGNMENTS = [
  alignments_with((1,  1));
  alignments_with((1, -1));
  alignments_with((0,  1));
  alignments_with((1,  0))]

function alignment_value_for(g::Game, player, alignment)
  γ = 0.1
  N = 0
  for pos in alignment
    cell = g.board[pos...]
    if cell == player
      N += 1
    elseif cell == other(player)
      return 0.
    end
  end
  return γ ^ (TO_CONNECT - 1 - N)
end

function heuristic_value_for(g::Game, player)
  return sum(alignment_value_for(g, player, al) for al in ALIGNMENTS)
end

function GI.heuristic_value(g::Game)
  mine = heuristic_value_for(g, g.curplayer)
  yours = heuristic_value_for(g, other(g.curplayer))
  return mine - yours
end

#####
##### ML interface
#####

flip_cell_color(c::Cell) = c == EMPTY ? EMPTY : other(c)

function flip_colors(board)
  return @SMatrix [
    flip_cell_color(board[col, row])
    for col in 1:NUM_COLS, row in 1:NUM_ROWS]
end

function GI.vectorize_state(::Type{Game}, state)
  board = GI.white_playing(Game, state) ? state.board : flip_colors(state.board)
  return Float32[
    board[col, row] == c
    for col in 1:NUM_COLS,
        row in 1:NUM_ROWS,
        c in [EMPTY, WHITE, BLACK]]
end

#####
##### Symmetries
#####

function flipped_board(board)
  return @SMatrix[board[col, row]
    for col in reverse(1:NUM_COLS), row in 1:NUM_ROWS]
end

function GI.symmetries(::Type{Game}, state)
  symb = flipped_board(state.board)
  σ = reverse(collect(1:NUM_COLS))
  syms = (board=Board(symb), curplayer=state.curplayer)
  return [(syms, σ)]
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

function GI.render(g::Game; with_position_names=true, botmargin=true)
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
      curplayer = WHITE
    elseif nw == nb + 1
      curplayer = BLACK
    else
      return nothing
    end
    return (board=board, curplayer=curplayer)
  catch e
    return nothing
  end
end
