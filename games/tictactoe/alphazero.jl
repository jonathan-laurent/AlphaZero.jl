using Revise
import AlphaZero

DIR = "session-tictactoe"

Revise.includet("game.jl") ; import .TicTacToe ; Game = TicTacToe.Game
Revise.includet("params.jl")

session = AlphaZero.Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

AlphaZero.resume!(session)

AlphaZero.explore(session)
