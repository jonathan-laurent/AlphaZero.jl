using Revise
import AlphaZero

Revise.includet("game.jl")
import .TicTacToe
Game = TicTacToe.Game
Revise.includet("params_debug.jl")

DIR = "session"

session = AlphaZero.Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

AlphaZero.resume!(session)

AlphaZero.plot_report(DIR)

# AlphaZero.resume!(session)

# AlphaZero.validate(Game, Network, DIR, validation)

# AlphaZero.explore(session)
