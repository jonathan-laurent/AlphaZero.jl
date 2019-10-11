import AlphaZero
import AlphaZero.GI

include("../games/tictactoe/game.jl")
include("../games/tictactoe/params.jl")

import .TicTacToe
const Game = TicTacToe.Game

network = AlphaZero.SimpleNet{Game, netparams}()
env = AlphaZero.Env{Game}(params, network)

using Profile
using ProfileView

AlphaZero.self_play!(env) # To compile every function
Profile.clear()
@profile AlphaZero.self_play!(env)
ProfileView.svgwrite("self-play-profile.svg")

# Last time I checked, the bottleneck was game simulation speed!
