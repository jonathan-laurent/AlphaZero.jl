#####
##### Evaluating the mistake rate of an AlphaZero agent
##### on the standard benchmark from Pascal Pons
#####

const SESSION_DIR = "sessions/connect-four"
const SAVE_FILE = "pons-benchmark-results.json"
const PLOT_FILE = "pons-benchmark-results.png"
const ITC_STRIDE = 5
const DEBUG_MODE = false # Launches a quick run on a tiny dataset to help debug

# A benchmark to evaluate connect-four agents is available at:
#   http://blog.gamesolver.org/solving-connect-four/02-test-protocol/
# It features 6000 positions along with their expected scores.

# Let `ne` the number of elapsed moves. A game stage is either
# `begin` (ne <= 14), `middle` (14 < ne <= 28) or `end` (ne > 28)
const STAGES = [:beginning, :middle, :end]

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

struct Bench
  stage :: Symbol
  difficulty :: Symbol
  entries :: Vector{Tuple{String, Int}}
end

function load_benchmarks(dir)
  @assert isdir(dir)
  benchmarks = Bench[]
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
    push!(benchmarks, Bench(stage, difficulty, entries))
    end
  end
  rank(b) = (
    findfirst(==(b.difficulty), DIFFICULTIES),
    findfirst(==(b.stage), STAGES))
  sort!(benchmarks, by=rank)
  return benchmarks
end

const BENCHMARKS_DIR = joinpath(@__DIR__, "..", "benchmark")
const BENCHMARKS = load_benchmarks(BENCHMARKS_DIR)

####
#### Testing code
####

using AlphaZero
include("../main.jl")
using .ConnectFour: Game, Solver, Training
using ProgressMeter
using Formatting

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
  turn = 1 # We don't bother computing a meaningful value as it is not used.
  as, π = AlphaZero.think(player, s, turn)
  api = argmax(π)
  qs = [Solver.qvalue(solver, s, a) for a in as]
  return sign(qs[api]) == maximum(sign, qs)
end

function test_player_on(player, benchmark, progress)
  solver = Solver.Player()
  nerr = 0
  entries = benchmark.entries
  DEBUG_MODE && (entries = Iterators.take(entries, 10))
  for e in entries
    optimal_on(solver, player, e) || (nerr += 1)
    next!(progress)
  end
  return nerr / length(entries)
end

function test_player(player)
  errs = Float64[]
  for bench in BENCHMARKS
    p = Progress(length(bench.entries))
    err = test_player_on(player, bench, p)
    push!(errs, err)
    err_str =
    println("($(bench.stage), $(bench.difficulty)): $(fmt(".2f", 100 * err))%")
  end
  return errs
end

function load_alphazero_player(dir)
  env = AlphaZero.UserInterface.load_env(
    Game, Training.Network{Game},
    AlphaZero.Log.Logger(devnull), dir, params=Training.params)
  mcts_params = MctsParams(env.params.arena.mcts,
    temperature=StepSchedule(0.),
    dirichlet_noise_ϵ=0.)
  return MctsPlayer(env.bestnn, mcts_params)
end

####
#### Running all the tests
####

const Errors = Vector{Float64} # One error rate for each benchmark

# Type gathering all produced data.
struct Results
  minmax :: Errors # Minmax baseline
  alphazero :: Errors #
  alphazero_training :: Vector{Tuple{Int, Errors}}
end

function generate_data(session_dir)
  println("Testing the minmax baseline")
  player = MinMax.Player{Game}(depth=5, τ=0)
  minmax_errs = test_player(player)
  println("")

  println("Testing AlphaZero")
  player = load_alphazero_player(session_dir)
  az_errs = test_player(player)
  println("")

  println("Testing AlphaZero during training")
  az_training_errs = []
  itc = 0
  while true
    itdir = joinpath(session_dir, "iterations", "$itc")
    AlphaZero.UserInterface.valid_session_dir(itdir) || break
    DEBUG_MODE && itc > 10 && break
    println("Iteration $itc")
    player = load_alphazero_player(itdir)
    push!(az_training_errs, (itc, test_player(player)))
    itc += ITC_STRIDE
  end
  return Results(minmax_errs, az_errs, az_training_errs)
end

#####
##### Main
#####

using JSON2
using JSON3

JSON3.StructType(::Type{Results}) = JSON3.Struct()

# Regenerate data only if needed as it can take a while.
if isfile(SAVE_FILE)
  data = open(SAVE_FILE, "r") do io
    JSON3.read(io, Results)
  end
else
  data = generate_data(SESSION_DIR)
  open(SAVE_FILE, "w") do io
    JSON2.pretty(io, JSON3.write(data))
  end
end

import Plots

# i is the benchmark number
function sub_plot(benchs, results, n)
  bench = benchs[n]
  sname(s) = uppercase(string(s)[1]) * string(s)[2:end]
  title = "$(sname(bench.stage)) - $(sname(bench.difficulty))"
  azt = results.alphazero_training
  xs = [itc for (itc, _) in azt]
  ys = [100 * errs[n] for (itc, errs) in azt]
  #ymax = max(5, maximum(ys))
  ymax = maximum(ys)
  plt = Plots.plot(
    xs, ys, ylims=(0, ymax),
    # xlabel="Training Iteration",
    ylabel="Error Rate (in %)",
    legend=nothing,
    title=title,
    titlefontsize=10,
    xguidefontsize=8,
    yguidefontsize=8)
  Plots.hline!(plt, [100 * results.minmax[n]])
  return plt
end

function plot_results(results)
  plts = [sub_plot(BENCHMARKS, results, n) for n in eachindex(BENCHMARKS)]
  return Plots.plot(plts..., layout=(3, 2), dpi=200)
end

plot = plot_results(data)
Plots.savefig(plot, PLOT_FILE)
