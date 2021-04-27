#####
##### Profiling backpropagation
#####

using AlphaZero
using Setfield
import CUDA

# nsys launch julia --project
# using Revise; Revise.includet("scripts/profile/backprop.jl")
# profile_backprop()
# nsight-sys ...

# Takeaway: time spent on GC skyrockets when you approach the memory limit

function profile_backprop(
    exp::Experiment = Examples.experiments["connect-four"];
    profile=false,
    num_games=5000,
    num_batches=100,
    batch_size=1024,
    num_filters=64)

  exp = @set exp.params.learning.batch_size = batch_size
  exp = @set exp.netparams.num_filters = num_filters
  session = Session(exp, autosave=false, dir="sessions/profile-backprop-$(exp.name)")
  env = session.env
  for i in 1:num_games
    trace = play_game(exp.gspec, RandomPlayer())
    AlphaZero.push_trace!(session.env.memory, trace, 1.0)
  end
  nparams = Network.num_parameters(env.curnn)
  UI.Log.section(session.logger, 1, "Profiling backprop")
  experience = AlphaZero.get_experience(env.memory)
  tr = AlphaZero.Trainer(env.gspec, env.curnn, experience, env.params.learning)
  AlphaZero.batch_updates!(tr, 1)  # to compile everything
  if profile
    CUDA.@profile AlphaZero.batch_updates!(tr, num_batches)
  else
    CUDA.@time AlphaZero.batch_updates!(tr, num_batches)
  end
  return
end

profile_backprop()