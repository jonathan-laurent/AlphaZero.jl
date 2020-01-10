#####
##### Generate a flame graph for self-play using ProfileView
#####

import AlphaZero

ENV["TRAINING_MODE"] = "debug"

include("../games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

network = Training.Network{Game}(Training.netparams)
env = AlphaZero.Env{Game}(Training.params, network)

using Profile
using ProfileView

# The neural network should not be used at iteration 0
AlphaZero.self_play!(env, nothing) # To compile every function
Profile.clear()
@profile AlphaZero.self_play!(env, nothing)
ProfileView.svgwrite("self-play-profile.svg")
