import AlphaZero
import AlphaZero.GI

include("../games/tictactoe/game.jl")
import .TicTacToe
Game = TicTacToe.Game
include("../games/tictactoe/params_debug.jl")


network = AlphaZero.SimpleNet{Game}(netparams)
env = AlphaZero.Env{Game}(params, network)

using Profile
#using ProfileView
using StatProfilerHTML

AlphaZero.self_play!(env, nothing) # To compile every function
Profile.clear()
@profilehtml AlphaZero.self_play!(env, nothing)
#open("self-play-profile.txt", "w") do io
#  Profile.print(io)
#send
#ProfileView.svgwrite("self-play-profile.svg")

# Last time I checked, the bottleneck was game simulation speed!
