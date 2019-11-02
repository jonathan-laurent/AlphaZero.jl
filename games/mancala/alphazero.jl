using Revise
using AlphaZero
using AlphaZero.FluxNets

Revise.includet("game.jl")
using .Mancala
Revise.includet("params.jl")

DIR = "session-mancala"

#=
function config(nblocks, nfilters, niters, nworkers)
  title = "$nblocks blocks, $nfilters filters, $niters iters, $nworkers workers"
  network = ResNet{Game}(ResNetHP(netparams,
    num_blocks=nblocks, num_filters=nfilters))
  params = MctsParams(self_play.mcts,
    num_iters_per_turn=niters,
    num_workers=nworkers)
  return (title, network, params)
end

AlphaZero.profile_self_play([
  config(5, 64, 640, 32),
  config(5, 64, 320, 32),
  config(5, 64, 160, 32)])
=#

session = Session(
  Game, Network, params, netparams,
  dir=DIR, autosave=true, validation=validation)

resume!(session)

explore(session)
