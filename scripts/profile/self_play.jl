#####
##### Profiling Self-play
#####

# using Revise; Revise.includet("scripts/profile/self_play.jl")
# profile_self_play()

# Note: when using the chrome_tracing logger, it looks like debug statements
# from GPUCompiler are shown that are normally hidden. This may be a bug
# in Logging or GPUCompiler and we should investigate it at some point.

using AlphaZero
using Setfield

using Profile
using LoggingExtras
# using ProfileSVG
# using ProfileView
# import StatProfilerHTML

import CUDA

function profile_self_play(
  exp::Experiment = Examples.experiments["connect-four"];
  num_games=512,
  num_workers=256,
  batch_size=128,
  num_filters=64)

  exp = @set exp.netparams.num_filters = num_filters
  exp = @set exp.params.self_play.sim.num_workers = num_workers
  exp = @set exp.params.self_play.sim.batch_size = batch_size
  exp = @set exp.params.self_play.sim.num_games = num_games

  session = Session(exp, autosave=false, dir="sessions/profile-backprop-$(exp.name)")
  env = session.env
  UI.Log.section(session.logger, 1, "Profiling data generation")
  CUDA.@time AlphaZero.self_play_step!(env, session)
  return
end

function full_profile()
  profile_self_play() # Compilation
  profile_self_play()
  profile_self_play() # Double-checking
end

function chrome_tracing()
  global_logger(TeeLogger(
    ConsoleLogger(),
    AlphaZero.ProfUtils.chrome_tracing_logger("tracing.json")))
  profile_self_play()
end

function flame_graph()
  profile_self_play() # Compilation
  Profile.init(n=10^8, delay=0.01)
  Profile.clear()
  @profile profile_self_play()
  ProfileSVG.save("self-play-profile.svg")
  # StatProfilerHTML.statprofilehtml()
  # ProfileView.view(C=true)
end

# full_profile()
chrome_tracing()

# Num cores experiments
# 1: 203s / 13% GC
# 2: 156s / 16% GC
# 3: 131s / 19% GC
# 4: 128s / 20% GC
# 6: 125s / 20% GC

# Knet vs Flux
# 1. Flux, 64 filters:  92 samples/s  (20% in GC)
# 2. Flux, 128 filters: 52 samples/s
# 3. Knet, 64 filters:  82 samples/s  (35% in GC)
# 4. Knet, 128 filters: 42 samples/s