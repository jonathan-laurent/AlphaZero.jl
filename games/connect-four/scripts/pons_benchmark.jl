#####
##### Evaluating the mistake rate of an AlphaZero agent
##### on the standard benchmark from Pascal Pons
#####

const SOLVER_ERROR =
  "To use this script, you should install Pascal Pons' Connect Four Solver.\n" *
  "See games/connect-four/solver/README.md for instructions."

const SESSION_DIR = "sessions/connect-four"
const SAVE_FILE = "pons-benchmark-results.json"
const PLOT_FILE = "pons-benchmark-results.png"
const ITC_MAX = 15  # Number of iterations taken into account in learning curves
const ITC_STRIDE = 1  # A benchmark is run every ITC_STRIDE iteration
const NUM_WORKERS = 128
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
using .Examples.ConnectFour: GameSpec, Solver, Training
using ProgressMeter
using Formatting
using Statistics: mean

const gspec = GameSpec()

function state_of_string(str)
  g = GI.init(gspec)
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

# TODO: the Solver is not thread safe.
function test_player_on(make_player, oracle, benchmark, progress)
  # Split benchmark entries into batches for workers
  entries = DEBUG_MODE ?
    Iterators.take(benchmark.entries, 2) :
    benchmark.entries
  batch_size = max(length(entries) ÷ NUM_WORKERS, 1)
  batches = Iterators.partition(entries, batch_size) |> collect
  nworkers = length(batches)
  # Spawn nworkers workers (nworkers <= NUM_WORKERS)
  spawn_oracle, done = AlphaZero.batchify_oracles(
    oracle; num_workers=nworkers, batch_size=nworkers, fill_batches=false)
  solver = Solver.Player()
  results = AlphaZero.Util.tmap_bg(batches) do batch
    player = make_player(spawn_oracle())
    res = map(batch) do e
      err = optimal_on(solver, player, e) ? 0. : 1.
      next!(progress)
      return err
    end
    done()
    return res
  end
  return mean(Iterators.flatten(results) |> collect)
end

function test_player(make_player, oracle)
  errs = Float64[]
  for bench in BENCHMARKS
    p = Progress(length(bench.entries))
    err = test_player_on(make_player, oracle, bench, p)
    push!(errs, err)
    err_str =
    println("($(bench.stage), $(bench.difficulty)): $(fmt(".2f", 100 * err))%")
  end
  return errs
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

function test_alphazero(env)
  mcts_params = MctsParams(env.params.arena.mcts,
    temperature=ConstSchedule(0.),
    dirichlet_noise_ϵ=0.)
  net = Network.copy(env.bestnn, on_gpu=true, test_mode=true)
  return test_player(net) do net
    return MctsPlayer(gspec, net, mcts_params)
  end
end

function generate_data(session_dir)
  println("Testing the minmax baseline")
  minmax_errs = test_player(nothing) do _
     MinMax.Player(depth=5, amplify_rewards=true, τ=0)
  end
  println("")

  println("Testing AlphaZero")
  env = AlphaZero.UserInterface.load_env(session_dir)
  az_errs = test_alphazero(env)
  println("")

  println("Testing AlphaZero during training")
  az_training_errs = []
  itc = 0
  while true
    itdir = joinpath(session_dir, "iterations", "$itc")
    if !AlphaZero.UserInterface.valid_session_dir(itdir)
      if isdir(itdir)
        error("""
        The directory '$itdir' does not contain a valid environment.
        Did you run the training session using `save_intermediate=true`?""")
      end
      break
    end
    DEBUG_MODE && itc > 10 && break
    itc > ITC_MAX && break
    println("Iteration $itc")
    env = AlphaZero.UserInterface.load_env(itdir)
    push!(az_training_errs, (itc, test_alphazero(env)))
    itc += ITC_STRIDE
  end
  return Results(minmax_errs, az_errs, az_training_errs)
end

#####
##### Main
#####

if !isdir(Solver.DEFAULT_SOLVER_DIR)
  println(stderr, SOLVER_ERROR)
  exit()
end

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
    JSON3.pretty(io, JSON3.write(data))
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
  y_minmax = 100 * results.minmax[n]
  #ymax = max(5, maximum(ys))
  ymax = max(maximum(ys), y_minmax)
  plt = Plots.plot(
    xs, ys, ylims=(0, ymax),
    # xlabel="Training Iteration",
    ylabel="Error Rate (in %)",
    legend=nothing,
    title=title,
    titlefontsize=10,
    xguidefontsize=8,
    yguidefontsize=8)
  Plots.hline!(plt, [y_minmax])
  return plt
end

function plot_results(results)
  plts = [sub_plot(BENCHMARKS, results, n) for n in eachindex(BENCHMARKS)]
  return Plots.plot(plts..., layout=(3, 2))
end

plot = plot_results(data)
Plots.savefig(plot, PLOT_FILE)
