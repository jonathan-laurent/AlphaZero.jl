#####
##### Interface to Pascal Pons' Connect 4 Solver
##### https://github.com/PascalPons/connect4
#####

# Problem: no Connect4 module. We can change this:

module Solver

import ..Game, ..history, ..WHITE, ..NUM_CELLS
import AlphaZero: GI, GameInterface, Benchmark, AbstractPlayer, think

const DEFAULT_SOLVER_DIR = joinpath(@__DIR__, "solver", "connect4")

struct Player <: AbstractPlayer{Game}
  process :: Base.Process
  function Player(;
      solver_dir=DEFAULT_SOLVER_DIR,
      solver_name="c4solver",
      disable_stderr=true)
    cmd = Cmd(`./$solver_name`, dir=solver_dir)
    if disable_stderr
      cmd = pipeline(cmd, stderr=devnull)
    end
    p = open(cmd, "r+")
    return new(p)
  end
end

# Solver protocol
# - Standard input: one position per line
# - Standard output: space separated
#     position, score, number of explored node, computation time in μs

struct SolverOutput
  score :: Int
  num_explored_nodes :: Int64
  time :: Int64 # in μs
end

history_string(game) = reduce(*, map(string, history(game)))

function query_solver(p::Player, g)
  hstr = history_string(g)
  println(p.process, hstr)
  l = readline(p.process)
  args = map(split(l)[2:end]) do x
    parse(Int64, x)
  end
  return SolverOutput(args...)
end

function remaining_stones(game, player)
  @assert !isnothing(game.history)
  n = length(game.history)
  p = n ÷ 2
  (n % 2 == 1 && player == WHITE) && (p += 1)
  return NUM_CELLS ÷ 2 - p
end

function value(player, game)
  z = GI.terminal_white_reward(game)
  if isnothing(z)
    return query_solver(player, game).score
  elseif iszero(z)
    return 0
  else
    v = remaining_stones(game, game.winner) + 1
    z < 0 && (v = -v)
    GI.white_playing(game) || (v = -v)
    return v
  end
end

function qvalue(player, game, action)
  @assert isnothing(GI.terminal_white_reward(game))
  wp = GI.white_playing(game)
  game = copy(game)
  GI.play!(game, action)
  pswitch = wp != GI.white_playing(game)
  nextv = value(player, game)
  return pswitch ? - nextv : nextv
end

function think(p::Player, g)
  as = GI.available_actions(g)
  qs = [qvalue(p, g, a) for a in as]
  maxq = maximum(qs)
  opt = findall(>=(maxq), qs)
  π = zeros(length(as))
  π[opt] .= 1 / length(opt)
  return as, π
end

Benchmark.PerfectPlayer(::Type{Game}) = Player

end
