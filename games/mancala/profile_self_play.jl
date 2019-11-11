using Revise
using AlphaZero

Revise.includet("game.jl")
using .Mancala
Revise.includet("params.jl")

function config(nblocks, nfilters, niters, nworkers)
  title = "$nblocks blocks, $nfilters filters, $niters iters, $nworkers workers"
  network = ResNet{Game}(ResNetHP(netparams,
    num_blocks=nblocks, num_filters=nfilters))
  params = MctsParams(self_play.mcts,
    num_iters_per_turn=niters,
    num_workers=nworkers)
  return (title, network, params)
end

# We want 3000 games per iteration?
# Iteration: self-play=30minutes
# So we want to generate
# We want to simulate 100 games per minute

# 2000 games, then iterate

nblocks = 10
nfilters = 128

AlphaZero.profile_self_play([
  config(7, 128, 320, 64),
  config(5, 128, 320, 64),
  config(5, 100, 320, 64),
  config(5, 64, 320, 64),
  config(10, 128, 1024, 256),
  config(5, 64, 1024, 256),
  config(5, 64, 160, 32),
])
