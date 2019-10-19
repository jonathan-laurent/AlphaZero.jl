using Revise
import AlphaZero

DIR = "session-mancala"

Revise.includet("game.jl") ; import .Mancala ; Game = Mancala.Game
Revise.includet("params_debug.jl")

session = AlphaZero.Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=false, validation=validation)

AlphaZero.resume!(session)

AlphaZero.explore(session)
