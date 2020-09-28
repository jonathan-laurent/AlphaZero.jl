#####
##### Generate a flame graph for self-play using ProfileSVG
#####

using AlphaZero

include("../games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: GameSpec, Training

include("../lib/dummy_run.jl")

gspec = GameSpec()
params, _ = dummy_run_params(Training.params, Training.benchmark)
network = Training.Network(gspec, Training.netparams)
env = AlphaZero.Env(gspec, params, network)

using Profile
using ProfileSVG

AlphaZero.self_play_step!(env, nothing) # To compile every function
Profile.clear()
@profile AlphaZero.self_play_step!(env, nothing)
ProfileSVG.save("self-play-profile.svg")
