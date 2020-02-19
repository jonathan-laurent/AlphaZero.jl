#####
##### Benchmark from Pascal Pons
#####

# A benchmark to evaluate connect-four agents is available at:
#     http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
# It features 6000 positions along with their expected scores.

# Let `ne` the number of elapsed moves. A game stage is either
# `begin` (ne <= 14), `middle` (14 < ne <= 28) or `end` (ne > 28)
const STAGES = [:begin, :middle, :end]

# Let `nr` the number of remaining moves. A difficulty level is either
# `easy` (nr < 14), `medium` (14 <= nr < 28) or `hard` (nr > 28).
const DIFFICULTIES = [:easy, :medium, :hard]

# Return either `nothing` or a `(stage, level)` tuple
function parse_test_filename(name)
  try
    m = match(r"^Test_L(\d)_R(\d)$", name)
    L = parse(Int, m.captures[1])
    R = parse(Int, m.captures[2])
    return (STAGES[L], DIFFICULTIES[R])
  catch e
    rethrow(e)
    return nothing
  end
end

struct Benchmark
  stage :: Symbol
  difficulty :: Symbol
  entries :: Vector{Tuple{String, Int}}
end

function load_benchmarks(dir)
  @assert isdir(dir)
  benchmarks = Benchmark[]
  files = readdir(dir)
  for bf in files
    meta = parse_test_filename(bf)
    f = joinpath(dir, bf)
    if !isnothing(meta) && isfile(f)
      stage, difficulty = meta
      entries = []
      for L in readlines(f)
        try
          L = split(L)
          push!(entries, (L[1], parse(Int, L[2])))
        catch e
          rethrow(e)
        end
      end
    push!(benchmarks, Benchmark(stage, difficulty, entries))
    end
  end
  return benchmarks
end

const BENCHMARKS_DIR = joinpath(@__DIR__, "..", "benchmark")
const BENCHMARKS = load_benchmarks(BENCHMARKS_DIR)

####
#### Load a session and run the test
####

using AlphaZero
include("../main.jl")
using .ConnectFour
import .ConnectFour: Solver

using ProgressMeter

function state_of_string(str)
  g = Game()
  for c in str
    a = parse(Int, c)
    GI.play!(g, a)
  end
  return g
end

function optimal_on(solver, player, e)
  s = state_of_string(e[1])
  as, π = AlphaZero.think(player, s)
  api = argmax(π)
  qs = [Solver.qvalue(solver, s, a) for a in as]
  return sign(qs[api]) == maximum(sign, qs)
end

function test_player_on(player, benchmark, progress)
  solver = Solver.Player()
  won = 0
  for e in benchmark.entries
    if optimal_on(solver, player, e)
      won += 1
    end
    next!(progress)
  end
  return won / length(benchmark.entries)
end

function test_player(player)
  for bench in BENCHMARKS
    p = Progress(length(bench.entries))
    score = test_player_on(player, bench, p)
    println("($(bench.stage), $(bench.difficulty)): $score")
  end
end

player = MinMax.Player{Game}(depth=5, τ=0)
#player = AlphaZero.RandomPlayer{Game}()
test_player(player)


#= Scores for minmax (depth 5)
(begin, easy):     0.983
(begin, medium):   0.927
(begin, hard):     0.766
(middle, easy):    0.993
(middle, medium):  0.908
(end, easy):       0.986
=#
