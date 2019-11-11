#####
##### Test a learning iteration on a precomputed experience buffer
##### To watch GPU memory consumption, use `nvidia-smi -l 1`
#####

using Revise
using AlphaZero

Revise.includet("game.jl")
using .Mancala
Revise.includet("params.jl")

DIR = "session-mancala-bug"
MEMFILE = joinpath(DIR, "mem.data")
using Serialization: serialize, deserialize

if !isfile(MEMFILE)
  # Generate the data
  mkpath(DIR)
  let session = Session(Game, Network,
      params, netparams, dir=DIR, autosave=false)
    AlphaZero.Log.section(session.logger, 1, "Generating playing experience")
    self_play!(session.env, session)
    serialize(MEMFILE, get_experience(session.env))
  end
end
experience = deserialize(MEMFILE)

network = Network(netparams)
env = Env{Game}(params, network, experience)
dir = joinpath(DIR, "session")
mkpath(dir)
session = Session(env, dir)
memory_report(session.env, session)
report = learning!(session.env, session)
