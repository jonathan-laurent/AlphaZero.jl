using Revise
using AlphaZero
using ProgressMeter
using Plots

# It is important not to profile async mcts on tic tac toe as the number
# of states in this game is small and so the mcts tree quickly contains
# every state.
include("../using_game.jl")
@using_game "connect-four"

REP = 500
NUM_ITERATIONS = 512
MAX_LOG_NWORKERS = 9 # 2^9 = 512

TIME_FIG = "mcts_speed"
INFERENCE_TIME_RATIO_FIG = "inference_time_ratio"

network = SimpleNet{Game}(SimpleNetHP(
  width=500,
  depth_common=4))

function profile(nworkers, ngames)
  mcts = MCTS.Env{Game}(network, nworkers=nworkers, cpuct=1.)
  time = @elapsed begin
    @showprogress for i in 1:ngames
      MCTS.explore!(mcts, Game(), NUM_ITERATIONS)
    end
  end
  return time, MCTS.inference_time_ratio(mcts)
end

println("Compile everything...")
profile(1, 1)
profile(2, 1)

results = []
ts = []
itrs = []
for i in 0:MAX_LOG_NWORKERS
  nworkers = 2 ^ i
  println("Profiling MCTS with $nworkers workers...")
  time, inference_time_ratio = profile(nworkers, REP)
  push!(ts, time)
  push!(itrs, inference_time_ratio)
end

xticks = (
  collect(0:MAX_LOG_NWORKERS),
  ["$(2^i)" for i in 0:MAX_LOG_NWORKERS])

plot(0:MAX_LOG_NWORKERS, ts[1] ./ ts,
  title="Async MCTS Speedup",
  ylabel="Speedup",
  xlabel="Number of workers",
  ylims=(0, Inf),
  legend=:none,
  xticks=xticks)

hline!([1])

savefig(TIME_FIG)

plot(0:MAX_LOG_NWORKERS, 100 * itrs,
  title="Percentage of time spent in inference",
  ylims=(0, 100),
  xlabel="Number of workers",
  legend=:none,
  xticks=xticks)

savefig(INFERENCE_TIME_RATIO_FIG)
