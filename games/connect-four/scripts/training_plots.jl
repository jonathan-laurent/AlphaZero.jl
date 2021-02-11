####
#### Regenerate the training curves displayed on the tutorial and on the README
####

using AlphaZero
using JSON3

# Defining the benchmarks
Training = Examples.ConnectFour.Training
netonly = Benchmark.NetworkOnly(τ=0.5)
alphazero = Benchmark.Full(Training.arena.mcts)
minmax = Benchmark.MinMaxTS(depth=5, amplify_rewards=true, τ=0.2)
mcts = Training.mcts_baseline
alphazero_benchmark = [
  Benchmark.Duel(alphazero, mcts, Training.benchmark_sim),
  Benchmark.Duel(alphazero, minmax, Training.benchmark_sim)]
netonly_benchmark = [
  Benchmark.Duel(netonly, mcts, Training.benchmark_sim),
  Benchmark.Duel(netonly, minmax, Training.benchmark_sim)]
full_benchmark = [alphazero_benchmark; netonly_benchmark]

dir = UserInterface.default_session_dir("connect-four")
logger = UserInterface.Log.Logger()

function rerun_benchmarks()
  UserInterface.run_new_benchmark(dir, "alphazero", alphazero_benchmark; logger)
  UserInterface.run_new_benchmark(dir, "netonly", netonly_benchmark; logger)
  UserInterface.run_new_benchmark(dir, "full", full_benchmark; logger)  
end

# Reuse the JSON benchmark data to make a different plot.
function replot_from_json(itcmax=nothing)
  repdir = joinpath(dir, "netonly")
  repfile = joinpath(repdir, UserInterface.BENCHMARK_FILE)
  reports = open(repfile, "r") do io
    JSON3.read(io, Vector{Report.Benchmark})
  end
  params = UserInterface.load_env(dir).params
  if !isnothing(itcmax)
    reports = reports[1:itcmax+1]
  end
  UserInterface.plot_benchmark(params, reports, repdir)
end