ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # "binned" / "split"

using AlphaZero
using AlphaZero.Log
using Formatting

const REPEAT = 200

#####
##### Profiling utilities
#####

function profile_self_play(network, params, n)
  player = AlphaZero.MctsPlayer(network, params)
  time = @elapsed for i in 1:n
    AlphaZero.play_game(player, player)
  end
  itr = MCTS.inference_time_ratio(player.mcts)
  aed = MCTS.average_exploration_depth(player.mcts)
  return (time / n, itr, aed)
end

function profile_self_play(configs::Vector)
  log  = Logger()
  time = Log.ColType(10, x -> format("{:.1f} min", x * 100 / 60))
  itr  = Log.ColType(5,  x -> format("{}%", round(Int, x * 100)))
  expd = Log.ColType(5,  x -> format("{:.1f}", x))
  tab  = Log.Table([
    ("T100", time, s -> s[1]),
    ("ITR",  itr,  s -> s[2]),
    ("EXPD", expd, s -> s[3])])
  for (title, net, params) in configs
    profile_self_play(net, params, 1) # Compilation
    rep = profile_self_play(net, params, REPEAT)
    Log.table_row(log, tab, rep, [title])
  end
end

#####
##### Main
#####

include("../games.jl")
const GAME = "connect-four"
const SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game, Training

function config(nblocks, nfilters, niters, nworkers)
  title = "$nblocks blocks, $nfilters filters, $niters iters, $nworkers workers"
  network = ResNet{Game}(ResNetHP(Training.netparams,
    num_blocks=nblocks, num_filters=nfilters))
  params = MctsParams(Training.params.self_play.mcts,
    num_iters_per_turn=niters)
  return (title, network, params)
end

profile_self_play([
  config( 7,  64, 400, 64),
  config( 7, 128, 400, 64),
  config(10, 128, 400, 64),
  config(10, 128, 320, 64),
])
