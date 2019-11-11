using Revise
using AlphaZero

Revise.includet("game.jl")
using .Mancala
Revise.includet("params.jl")

DIR = "session-mancala"

session = Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=nothing)

resume!(session)

explore(session)
