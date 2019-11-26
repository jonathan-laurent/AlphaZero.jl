#####
##### Test a learning iteration on a precomputed experience buffer
##### To watch GPU memory consumption, use `nvidia-smi -l 1`
#####

ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000

using Revise
using AlphaZero

include("using_game.jl")
@using_default_game

DIR = joinpath("sessions", "test-learning", GAME)
MEMFILE = joinpath(DIR, "mem.data")
using Serialization: serialize, deserialize

if !isfile(MEMFILE)
  # Generate the data
  mkpath(DIR)
  let session = Session(Game, Net,
      params, netparams, dir=DIR, autosave=false)
    AlphaZero.Log.section(session.logger, 1, "Generating playing experience")
    self_play!(session.env, session)
    serialize(MEMFILE, get_experience(session.env))
  end
end
experience = deserialize(MEMFILE)

network = Net(netparams)
env = Env{Game}(params, network, experience)
dir = joinpath(DIR, "session")
mkpath(dir)
session = Session(env, dir)
#memory_report(session.env, session)
report = learning!(session.env, session)
