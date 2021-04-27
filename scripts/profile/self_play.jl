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

profile_self_play() # Compilation
profile_self_play()
profile_self_play() # Double-checking