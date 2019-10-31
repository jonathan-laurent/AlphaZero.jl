using Revise
using AlphaZero
using AlphaZero.FluxNets

DIR = "session-tictactoe"

Revise.includet("game.jl") ; import .TicTacToe ; Game = TicTacToe.Game
Revise.includet("params.jl")

session = Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

resume!(session)

explore(session)
