#####
##### Generate a flame graph for self-play using ProfileSVG
#####

using AlphaZero
using Profile
using ProfileSVG

experiment = Examples.experiments["tictactoe"] |> Scripts.dummy_run_experiment
network = experiment.mknet(experiment.gspec, experiment.netparams)
env = AlphaZero.Env(experiment.gspec, experiment.params, network)

AlphaZero.self_play_step!(env, nothing) # To compile every function
Profile.clear()
@profile AlphaZero.self_play_step!(env, nothing)
ProfileSVG.save("self-play-profile.svg")
