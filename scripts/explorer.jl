using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "tictactoe")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

const SESSION_DIR = joinpath("sessions", GAME)

session = Session(
  Game, Training.Network{Game},
  Training.params, Training.netparams, benchmark=Training.benchmark,
  dir=SESSION_DIR, autosave=false, save_intermediate=false)

explorer = Explorer(session.env)
explore!(explorer)
