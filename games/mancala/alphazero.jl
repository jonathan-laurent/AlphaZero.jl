using Revise
using AlphaZero
using AlphaZero.FluxNets

Revise.includet("game.jl")
using .Mancala
Revise.includet("params.jl")

DIR = "session-mancala"

session = Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

resume!(session)

explore(session)
