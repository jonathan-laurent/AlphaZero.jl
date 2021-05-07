import AlphaZero.GI

using Crayons
using StaticArrays

const NUM_HOUSES_PER_PLAYER = 6

const NUM_INITIAL_SEEDS_PER_HOUSE = 3

const NUM_HOUSES = 2 * NUM_HOUSES_PER_PLAYER

const NUM_SEEDS =  NUM_HOUSES * NUM_INITIAL_SEEDS_PER_HOUSE

const Player = Int
const WHITE = 1
const BLACK = 2

other(p::Player) = 3 - p

struct Board
  stores :: SVector{2, UInt8}
  houses :: SMatrix{2, NUM_HOUSES_PER_PLAYER, UInt8, NUM_HOUSES}
end

const INITIAL_BOARD =
  let stores = SVector(0, 0)
  let houses = @SMatrix[
      NUM_INITIAL_SEEDS_PER_HOUSE
      for i in 1:2, j in 1:NUM_HOUSES_PER_PLAYER]
    Board(stores, houses)
  end end

const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
  finished :: Bool
end

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  return GameEnv(board, curplayer, finished)
end

GI.spec(::GameEnv) = GameSpec()

GI.two_players(::GameSpec) = true

function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  if sum_houses(g.board, g.curplayer) == 0 || sum_houses(g.board, other(g.curplayer)) == 0
    g.finished = true
  end
end

const ACTIONS = collect(1:NUM_HOUSES_PER_PLAYER)
GI.actions(::GameSpec) = ACTIONS

#####
##### Position system to ease board manipulation
#####

struct HousePos
  player :: Int
  num :: Int
end

struct StorePos
  player :: Int
end

const Pos = Union{HousePos, StorePos}

function next_pos(pos::HousePos, player)
  if pos.num > 1
    return HousePos(pos.player, pos.num - 1)
  elseif pos.num == 1
    if pos.player == player
      return StorePos(player)
    else
      return HousePos(player, NUM_HOUSES_PER_PLAYER)
    end
  else
    @assert false
  end
end

function next_pos(pos::StorePos, player)
  @assert pos.player == player
  return HousePos(other(player), NUM_HOUSES_PER_PLAYER)
end

read_pos(b::Board, pos::HousePos) = b.houses[pos.player, pos.num]
read_pos(b::Board, pos::StorePos) = b.stores[pos.player]

function write_pos(b::Board, pos::HousePos, v)
  houses = setindex(b.houses, v, pos.player, pos.num)
  return Board(b.stores, houses)
end

function write_pos(b::Board, pos::StorePos, v)
  stores = setindex(b.stores, v, pos.player)
  return Board(stores, b.houses)
end

function opposite_pos(pos::HousePos)
  return HousePos(other(pos.player), NUM_HOUSES_PER_PLAYER - pos.num + 1)
end


#####
##### Defining game rules
#####

GI.actions_mask(g::GameEnv, player) = map(>(0), g.board.houses[player,:])

GI.actions_mask(g::GameEnv) = GI.actions_mask(g, g.curplayer)

function capture_last_and_opposite(b::Board, pos::HousePos)
  # @assert read_pos(b ,pos) == 1
  new_store = b.stores[pos.player] + read_pos(b, opposite_pos(pos)) + 1
  stores = setindex(b.stores, new_store, pos.player)
  b = write_pos(b, pos, 0)
  b = write_pos(b, opposite_pos(pos), 0)
  return Board(stores, b.houses)
end

sum_houses(b::Board, player) = sum(b.houses[player,:])

function capture_leftovers(b::Board, player)
  # @assert sum_houses(b, other(player)) == 0
  new_store = b.stores[player] + sum_houses(b, player)
  stores = setindex(b.stores, new_store, player)
  return Board(stores, zero(typeof(b.houses)))
end

# TODO tide up ending conditions
function GI.play!(g::GameEnv, a)
  pos = HousePos(g.curplayer, a)
  nseeds = read_pos(g.board, pos)
  @assert nseeds > 0
  g.board = write_pos(g.board, pos, 0)
  for i in 1:nseeds
    pos = next_pos(pos, g.curplayer)
    g.board = write_pos(g.board, pos, read_pos(g.board, pos) + 1)
  end
  # Check endgame
  if sum_houses(g.board, g.curplayer) == 0
    g.board = capture_leftovers(g.board, other(g.curplayer))
    g.finished = true
  elseif isa(pos, HousePos)
    # Capture opposite house if last seed was put in empty house on your side # MUST CHECK ENDGAME AFTER
    if read_pos(g.board, pos) == 1 && g.curplayer == pos.player
      g.board = capture_last_and_opposite(g.board, pos)
      # if captured last pieces of opposite player then game ends
      if sum_houses(g.board, other(g.curplayer)) == 0
        g.board = capture_leftovers(g.board, g.curplayer)
        g.finished = true
        return
      end
      if sum_houses(g.board, g.curplayer) == 0
        g.board = capture_leftovers(g.board, other(g.curplayer))
        g.finished = true
        return
      end
    end
    # Free turn if last seed was put in a store
    g.curplayer = other(g.curplayer)
  end
  return
end

GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

#####
##### Reward shaping
#####

function GI.game_terminated(g::GameEnv)
  return g.finished
end

function zero_one_reward(nw, nb)
  nw > nb && return 1.
  nw < nb && return -1.
  return 0.
end

linear_reward(nw, nb) = (nw - nb) / NUM_SEEDS

function GI.white_reward(g::GameEnv)
  if g.finished
    nw, nb = g.board.stores
    return zero_one_reward(nw, nb) # + linear_reward(nw, nb)
  else
    return 0.
  end
end

#####
##### Simple heuristic for minmax
#####

# This is an extremely naive heuristic and one can probably do better.
function GI.heuristic_value(g::GameEnv)
  nw, nb = g.board.stores
  v = nw - nb
  g.curplayer == BLACK && (v = -v)
  return Float64(v)
end

#####
##### ML interface
#####

function flip_colors(board)
  stores = @SVector [INITIAL_BOARD.stores[2],INITIAL_BOARD.stores[1]]
  houses = SMatrix{2, NUM_HOUSES_PER_PLAYER, UInt8, NUM_HOUSES}(
    [INITIAL_BOARD.houses[2,:]'; INITIAL_BOARD.houses[1,:]'])
  return Board(stores, houses)
end

function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  function cell(pos, chan)
    if chan == :nstones
      return read_pos(board, pos)
    elseif chan == :whouse
      return isa(pos, HousePos) && pos.player == WHITE
    elseif chan == :wstore
      return isa(pos, StorePos) && pos.player == WHITE
    elseif chan == :bhouse
      return isa(pos, HousePos) && pos.player == BLACK
    elseif chan == :bstore
      return isa(pos, StorePos) && pos.player == BLACK
    end
    @assert false
  end
  positions = [
    [HousePos(WHITE, i) for i in NUM_HOUSES_PER_PLAYER:-1:1];
    [StorePos(WHITE)];
    [HousePos(BLACK, i) for i in NUM_HOUSES_PER_PLAYER:-1:1];
    [StorePos(BLACK)]]
  return Float32[
    cell(p, c)
    for p in positions,
        y in 1:1,
        c in [:nstones, :whouse, :wstore, :bhouse, :bstore]]
end

#####
##### User interface
#####

GI.action_string(::GameSpec, a) = string(a)

function GI.parse_action(::GameSpec, str)
  try
    p = parse(Int, str)
    1 <= p <= NUM_HOUSES_PER_PLAYER ? p : nothing
  catch
    nothing
  end
end

#          1    2    3    4    5    6
#  +----+----+----+----+----+----+----+----+
#  |    |  3 |  3 |  3 |  3 |  3 |  3 |    |
#  |    +----+----+----+----+----+----+    |
#  |    |  3 |  3 |  3 |  3 |  3 |  3 |    |
#  +----+----+----+----+----+----+----+----+
#          6    5    4    3    2    1

function GI.render(g::GameEnv, with_position_names=true, botmargin=true)
  b = g.board
  gray(s...) = print(crayon"cyan", s..., crayon"reset")
  white(s...) = print(crayon"white", s..., crayon"reset")
  bold(s...) = print(crayon"yellow", s..., crayon"reset")
  blank_cell() = repeat(" ", 4)
  num_cell(n) = n == 0 ? blank_cell() : (n < 10 ? "  $n " : " $n ")
  function show_labels(labels)
    print(" "); print(blank_cell()); print(" ")
    for i in labels
      gray(num_cell(i)); print(" ")
    end
    print("\n")
  end
  function hline(len; newline=true)
    for i in 1:len
      gray("+----")
    end
    newline && gray("+\n")
  end
  function print_houses(houses, self)
    printing = self ? bold : white
    for v in houses
      gray("|"); printing(num_cell(v))
    end
  end
  function print_half(houses, self=false)
    gray("|"); print(blank_cell())
    print_houses(houses, self)
    gray("|"); print(blank_cell())
    gray("|\n")
  end
  function print_middle(stores)
    gray("|"); white(num_cell(stores[2]))
    hline(NUM_HOUSES_PER_PLAYER, newline=false)
    gray("+"); white(num_cell(stores[1]))
    gray("|\n")
  end
  show_labels(1:NUM_HOUSES_PER_PLAYER)
  hline(NUM_HOUSES_PER_PLAYER + 2)
  print_half(b.houses[2,:], g.curplayer == BLACK)
  print_middle(b.stores)
  print_half(reverse(b.houses[1,:]), g.curplayer == WHITE)
  hline(NUM_HOUSES_PER_PLAYER + 2)
  show_labels(reverse(1:NUM_HOUSES_PER_PLAYER))
  botmargin && print("\n")
end

function GI.read_state(::Type{GameEnv})
  try
    function read_houses(player)
      print("Player $(player) houses: ")
      houses = [parse(Int, s) for s in split(readline())]
      @assert all(>=(0), houses)
      @assert length(houses) == NUM_HOUSES_PER_PLAYER
      return houses
    end
    function read_store(player)
      print("Player $(player) store: ")
      store = parse(Int, readline())
      @assert store >= 0
      return store
    end
    # Read board
    h1 = read_houses(1)
    h2 = read_houses(2)
    s1 = read_store(1)
    #s2 = read_store(2)
    nsofar = sum(h1) + sum(h2) + s1
    @assert nsofar <= NUM_SEEDS
    s2 = NUM_SEEDS - nsofar
    board = Board([s1, s2], [h1'; h2'])
    # Read current player
    print("Current player (1/2): ")
    curplayer = parse(Int, readline())
    @assert 1 <= curplayer <= 2
    return GameEnv(board, curplayer == WHITE)
  catch
    return nothing
  end
end

# Example of creating game objects manually:
# Game(Board([10, 20], [[1 0 0 0 0 0]; [1 0 2 0 2 0]]), true)
