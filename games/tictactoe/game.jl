import AlphaZero.GI
using StaticArrays

const BOARD_SIDE = 3
const NUM_POSITIONS = BOARD_SIDE ^ 2

const Player = Bool
const WHITE = true
const BLACK = false

const Cell = Union{Nothing, Player}
const Board = SVector{NUM_POSITIONS, Cell}
const INITIAL_BOARD = Board(repeat([nothing], NUM_POSITIONS))
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

# TODO: we could have the game parametrized by grid size.
struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
end

GI.init(::GameSpec, state=INITIAL_STATE) = GameEnv(state.board, state.curplayer)

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = true

function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
end

#####
##### Defining winning conditions
#####

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

function has_won(g::GameEnv, player)
  any(ALIGNMENTS) do al
    all(al) do pos
      g.board[pos] == player
    end
  end
end

#####
##### Game API
#####

const ACTIONS = collect(1:NUM_POSITIONS)

GI.actions(::GameSpec) = ACTIONS

GI.actions_mask(g::GameEnv) = map(isnothing, g.board)

GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)

GI.white_playing(g::GameEnv) = g.curplayer

function terminal_white_reward(g::GameEnv)
  has_won(g, WHITE) && return 1.
  has_won(g, BLACK) && return -1.
  isempty(GI.available_actions(g)) && return 0.
  return nothing
end

GI.game_terminated(g::GameEnv) = !isnothing(terminal_white_reward(g))

function GI.white_reward(g::GameEnv)
  z = terminal_white_reward(g)
  return isnothing(z) ? 0. : z
end

function GI.play!(g::GameEnv, pos)
  g.board = setindex(g.board, g.curplayer, pos)
  g.curplayer = !g.curplayer
end

#####
##### Simple heuristic for minmax
#####

function alignment_value_for(g::GameEnv, player, alignment)
  γ = 0.3
  N = 0
  for pos in alignment
    mark = g.board[pos]
    if mark == player
      N += 1
    elseif !isnothing(mark)
      return 0.
    end
  end
  return γ ^ (BOARD_SIDE - 1 - N)
end

function heuristic_value_for(g::GameEnv, player)
  return sum(alignment_value_for(g, player, al) for al in ALIGNMENTS)
end

function GI.heuristic_value(g::GameEnv)
  mine = heuristic_value_for(g, g.curplayer)
  yours = heuristic_value_for(g, !g.curplayer)
  return mine - yours
end

#####
##### Machine Learning API
#####

function flip_colors(board)
  flip(cell) = isnothing(cell) ? nothing : !cell
  # Inference fails when using `map`
  return @SVector Cell[flip(board[i]) for i in 1:NUM_POSITIONS]
end

# Vectorized representation: 3x3x3 array
# Channels: free, white, black
# The board is represented from the perspective of white
# (as if white were to play next)
function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    board[pos_of_xy((x, y))] == c
    for x in 1:BOARD_SIDE,
        y in 1:BOARD_SIDE,
        c in [nothing, WHITE, BLACK]]
end

#####
##### Symmetries
#####

function generate_dihedral_symmetries()
  N = BOARD_SIDE
  rot((x, y)) = (y, N - x + 1) # 90° rotation
  flip((x, y)) = (x, N - y + 1) # flip along vertical axis
  ap(f) = p -> pos_of_xy(f(xy_of_pos(p)))
  sym(f) = map(ap(f), collect(1:NUM_POSITIONS))
  rot2 = rot ∘ rot
  rot3 = rot2 ∘ rot
  return [
    sym(rot), sym(rot2), sym(rot3),
    sym(flip), sym(flip ∘ rot), sym(flip ∘ rot2), sym(flip ∘ rot3)]
end

const SYMMETRIES = generate_dihedral_symmetries()

function GI.symmetries(::GameSpec, s)
  return [
    ((board=Board(s.board[sym]), curplayer=s.curplayer), sym)
    for sym in SYMMETRIES]
end

#####
##### Interaction API
#####

function GI.action_string(::GameSpec, a)
  string(Char(Int('A') + a - 1))
end

function GI.parse_action(::GameSpec, str)
  length(str) == 1 || (return nothing)
  x = Int(uppercase(str[1])) - Int('A')
  (0 <= x < NUM_POSITIONS) ? x + 1 : nothing
end

function read_board(::GameSpec)
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

function GI.read_state(::GameSpec)
  b = read_board(GameSpec())
  nw = count(==(WHITE), b)
  nb = count(==(BLACK), b)
  if nw == nb
    return (board=b, curplayer=WHITE)
  elseif nw == nb + 1
    return (board=b, curplayer=BLACK)
  else
    return nothing
  end
end

using Crayons

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red" : "Blue"
player_mark(p)  = p == WHITE ? "o" : "x"

function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
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
        print(GI.action_string(GI.spec(g), pos_of_xy((x, y))), " ")
      end
    end
    print("\n")
  end
  botmargin && print("\n")
end
