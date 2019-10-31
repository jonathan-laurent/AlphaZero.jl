using Revise
using AlphaZero
using AlphaZero.FluxNets

Revise.includet("game.jl")
using .TicTacToe
Revise.includet("params_debug.jl")

DIR = "session-tictactoe"

session = Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

resume!(session)

explore(session)
