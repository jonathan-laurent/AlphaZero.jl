#####
##### Testing distributed self-play
#####

ENV["JULIA_CUDA_MEMORY_POOL"] = "split"

using AlphaZero
using ProgressMeter

include("../../games/connect-four/main.jl")
using .ConnectFour: Game, Training

params = Training.params
network = Training.Network{Game}(Training.netparams)
env = AlphaZero.Env{Game}(params, network)

const sp_progress = Progress(params.self_play.num_games)
const arena_progress = Progress(params.arena.num_games)

struct Handler end
AlphaZero.Handlers.game_played(::Handler) = next!(sp_progress)
AlphaZero.Handlers.checkpoint_game_played(::Handler) = next!(arena_progress)

println("Running on $(Threads.nthreads()) threads.")

report, t, mem, gct = @timed AlphaZero.self_play_step!(env, Handler())

# (avgr, redundancy), t, mem, gct = @timed AlphaZero.evaluate_network(
#   env.curnn, env.bestnn, env.params, Handler())

println("Total time: $t")
println("Spent in GC: $gct")
