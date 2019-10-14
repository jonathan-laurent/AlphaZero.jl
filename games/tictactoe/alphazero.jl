using Revise
import AlphaZero

Revise.includet("game.jl")
import .TicTacToe
Game = TicTacToe.Game
Revise.includet("params.jl")

session = AlphaZero.Session(
  Game, Network, params, netparams,
  dir="session", autosave=true)

AlphaZero.resume!(session)
AlphaZero.explore(session)
