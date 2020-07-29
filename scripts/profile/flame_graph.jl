#####
##### Generate a flame graph for self-play using ProfileView
#####

using AlphaZero

include("../games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

include("../lib/dummy_run.jl")

params, _ = dummy_run_params(Training.params, Training.benchmark)
network = Training.Network{Game}(Training.netparams)
env = AlphaZero.Env{Game}(params, network)

using Profile
using ProfileSVG

# using Juno
# Juno.@enter AlphaZero.self_play_step!(env, nothing)

AlphaZero.self_play_worker(env.bestnn, env.params.self_play, ReentrantLock(), nothing , 1)

# The neural network should not be used at iteration 0
# AlphaZero.self_play_step!(env, nothing) # To compile every function
# Profile.clear()
# @profile AlphaZero.self_play_step!(env, nothing)
# ProfileSVG.save("self-play-profile.svg")
