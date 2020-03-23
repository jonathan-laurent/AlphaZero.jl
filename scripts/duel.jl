#####
##### Duel.jl
##### Simple script to run custom benchmark duels
#####

using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

SESSION_DIR = "sessions/$GAME"

AlphaZero.UserInterface.run_duel(Game, Training.Network{Game}, SESSION_DIR,
  Benchmark.Duel(
    Benchmark.NetworkOnly(),
    Benchmark.MinMaxTS(depth=5, τ=0.2),
    num_games=200,
    flip_probability=0.5,
    color_policy=CONTENDER_WHITE),
  params=Training.params)
