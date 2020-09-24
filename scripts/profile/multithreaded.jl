#####
##### Testing multithreaded self-play
#####

const NUM_GAMES = 500

ENV["JULIA_CUDA_MEMORY_POOL"] = "split"

# ENV["ALPHAZERO_DEFAULT_DL_FRAMEWORK"] = "FLUX" # FLUX / KNET

# When running on a CPU, having multiple threads does not play
# well with BLAS multithreading
using LinearAlgebra
BLAS.set_num_threads(1)

using AlphaZero
using ProgressMeter

include("../../games/connect-four/main.jl")
using .ConnectFour: GameSpec, Training

struct Handler
  progress :: Progress
  Handler(n) = new(Progress(n))
end
AlphaZero.Handlers.game_played(h::Handler) = next!(h.progress)
AlphaZero.Handlers.checkpoint_game_played(h::Handler) = next!(h.progress)

function bench_self_play(n)
  gspec = GameSpec()
  params = Training.params
  params = Params(params, self_play=SelfPlayParams(params.self_play, num_games=n))
  network = Training.Network(gspec, Training.netparams)
  env = AlphaZero.Env(gspec, params, network)
  return @timed AlphaZero.self_play_step!(env, Handler(n))
end

println("Running on $(Threads.nthreads()) threads.")
println("Playing one game to compile everything.")
bench_self_play(1)
println("Starting benchmark.")
report, t, mem, gct = bench_self_play(NUM_GAMES)

println("Total time: $t")
println("Spent in GC: $gct")