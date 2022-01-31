#####
##### Interface to Pascal Pons' Connect 4 Solver
##### https://github.com/PascalPons/connect4
#####

# TODO: at some point, I should use ConnectFourSolver.jl
# To represent state "031", we must do `p = Position(); p(0); p(3); p(1); p`.

module Solver

using AlphaZero

import ..GameEnv, ..history, ..WHITE, ..NUM_CELLS

const DEFAULT_SOLVER_DIR = joinpath(@__DIR__, "solver", "connect4")

struct Player <: AbstractPlayer
  process :: Base.Process
  lock :: ReentrantLock
  function Player(;
      solver_dir=DEFAULT_SOLVER_DIR,
      solver_name="c4solver",
      disable_stderr=true)
    cmd = Cmd(`./$solver_name`, dir=solver_dir)
    if disable_stderr
      cmd = pipeline(cmd, stderr=devnull)
    end
    p = open(cmd, "r+")
    return new(p, ReentrantLock())
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
  Base.lock(p.lock)
  println(p.process, hstr)
  l = readline(p.process)
  Base.unlock(p.lock)
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
  if !GI.game_terminated(game)
    return query_solver(player, game).score
  elseif game.winner == 0x00
    return 0
  else
    v = remaining_stones(game, game.winner) + 1
    if (game.winner == WHITE) != GI.white_playing(game)
      v = -v
    end
    return v
  end
end

function qvalue(player, game, action)
  @assert !GI.game_terminated(game)
  next = GI.clone(game)
  GI.play!(next, action)
  qnext = value(player, next)
  if GI.white_playing(game) != GI.white_playing(next)
    qnext = -qnext
  end
  return qnext
end

function AlphaZero.think(p::Player, g)
  as = GI.available_actions(g)
  qs = [qvalue(p, g, a) for a in as]
  maxq = maximum(qs)
  opt = findall(>=(maxq), qs)
  π = zeros(length(as))
  π[opt] .= 1 / length(opt)
  return as, π
end

end
