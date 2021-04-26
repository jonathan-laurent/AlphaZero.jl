#####
##### Profiling Self-play
#####

# using Revise; Revise.includet("scripts/profile/self_play.jl")
# profile_self_play()

using AlphaZero
using Setfield

import CUDA

function profile_self_play(
  exp::Experiment = Examples.experiments["connect-four"];
  num_games=512,
  num_workers=128,
  num_filters=64)
  
  exp = @set exp.netparams.num_filters = num_filters
  exp = @set exp.params.self_play.sim.num_workers = num_workers
  exp = @set exp.params.self_play.sim.num_games = num_games

  session = Session(exp, autosave=false, dir="sessions/profile-backprop-$(exp.name)")
  env = session.env
  UI.Log.section(session.logger, 1, "Profiling data generation")
  CUDA.@time AlphaZero.self_play_step!(env, session)
  return
end

profile_self_play() # Compilation
profile_self_play()
profile_self_play() # Double-checking


# BEFORE:
# 136.605654 seconds
#  (644.19 M CPU allocations: 42.165 GiB, 13.44% gc time) (4.27 M GPU allocations: 8.557 TiB, 26.02% gc time of which 25.24% spent allocating)
# 139.927391 seconds
#  (653.11 M CPU allocations: 43.247 GiB, 14.19% gc time) (4.28 M GPU allocations: 8.568 TiB, 26.36% gc time of which 24.51% spent allocating)

# AFTER main thread (no difference)
# 133.424587 seconds (571.59 M CPU allocations: 40.241 GiB, 13.96% gc time) (4.06 M GPU allocations: 8.131 TiB, 25.91% gc time of which 23.99% spent allocating)
# 144.442098 seconds (606.50 M CPU allocations: 42.566 GiB, 13.87% gc time) (4.38 M GPU allocations: 8.777 TiB, 25.83% gc time of which 24.19% spent allocating)
