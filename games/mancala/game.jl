module Mancala

export Game, Board

import AlphaZero.GI

using Printf
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

mutable struct Game
  board :: Board
  curplayer :: Player
  function Game(board::Board=INITIAL_BOARD, white_playing::Bool=true)
    new(board, white_playing ? WHITE : BLACK)
  end
end

GI.Board(::Type{Game}) = Board
GI.Action(::Type{Game}) = Int

Base.copy(g::Game) = Game(g.board, g.curplayer == WHITE)


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

function next_pos(pos, player)
  if isa(pos, HousePos)
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
  elseif isa(pos, StorePos)
    @assert pos.player == player
    return HousePos(other(player), NUM_HOUSES_PER_PLAYER)
  else
    @assert false
  end
end

function read_pos(b::Board, pos)
  isa(pos, HousePos) ?
    b.houses[pos.player, pos.num] :
    b.stores[pos.player]
end

function write_pos(b::Board, pos, v)
  if isa(pos, HousePos)
    houses = setindex(b.houses, v, pos.player, pos.num)
    return Board(b.stores, houses)
  else
    stores = setindex(b.stores, v, pos.player)
    return Board(stores, b.houses)
  end
end


#####
##### Defining game rules
#####

GI.available_actions(g::Game, player) = findall(>(0), g.board.houses[player,:])

GI.available_actions(g::Game) = GI.available_actions(g, g.curplayer)

sum_houses(b::Board, player) = sum(b.houses[player,:])

function GI.play!(g::Game, a)
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
    other_sum = sum_houses(g.board, other(g.curplayer))
    new_store = g.board.stores[g.curplayer] + other_sum
    stores = setindex(g.board.stores, new_store, g.curplayer)
    g.board = Board(stores, zero(typeof(g.board.houses)))
  # Free turn is last seed was put in a store
  elseif isa(pos, HousePos)
    g.curplayer = other(g.curplayer)
  end
  return
end

GI.board(g::Game) = g.board

function GI.board_symmetric(g::Game) :: Board
  b = g.board
  stores = @SVector[b.stores[2], b.stores[1]]
  houses = vcat(b.houses[2,:]', b.houses[1,:]')
  Board(stores, houses)
end

GI.white_playing(g::Game) = g.curplayer == WHITE

game_terminated(g::Game) = all(==(0), g.board.houses)


#####
##### Reward shaping
#####

function zero_one_reward(nw, nb)
  nw > nb && return 1.
  nw < nb && return -1.
  return 0.
end

linear_reward(nw, nb) = (nw - nb) / NUM_SEEDS

function GI.white_reward(g::Game)
  if game_terminated(g)
    nw, nb = g.board.stores
    return zero_one_reward(nw, nb) + linear_reward(nw, nb)
  else
    return nothing
  end
end


#####
##### ML interface
#####

GI.board_dim(::Type{Game}) = NUM_HOUSES + 2

GI.num_actions(::Type{Game}) = NUM_HOUSES_PER_PLAYER

GI.action_id(::Type{Game}, a) = a

GI.action(::Type{Game}, id) = id

function GI.vectorize_board(::Type{Game}, board)
  Float64[
    board.houses[1,:]; board.stores[1];
    board.houses[2,:]; board.stores[2]]
end


#####
##### User interface
#####

GI.action_string(::Type{Game}, a) = string(a)

function GI.parse_action(g::Game, str)
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

function GI.print_state(g::Game, with_position_names=true)
  b = g.board
  gray(s...) = print(crayon"blue", s..., crayon"reset")
  white(s...) = print(crayon"white", s..., crayon"reset")
  bold(s...) = print(crayon"yellow", s..., crayon"reset")
  blank_cell() = repeat(" ", 4)
  num_cell(n) = n == 0 ? blank_cell() : @sprintf(" %2d ", n)
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
end

function GI.read_state(::Type{Game})
  #try
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
    return Game(board, curplayer == WHITE)
  #catch
  #  return nothing
  #end
end

# Example of creating game objects manually:
# Game(Board([10, 20], [[1 0 0 0 0 0]; [1 0 2 0 2 0]]), true)

end
