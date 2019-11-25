ENV["CUARRAYS_MEMORY_LIMIT"] = 7_500_000_000
#ENV["CUARRAYS_MEMORY_POOL"] = "split" # "binned"

using Revise
using AlphaZero

include("using_game.jl")
@using_default_game

session = Session(
  Game, Network, params, netparams,
  dir=SESSION_DIR, autosave=true, validation=validation)

resume!(session)

# explore(session)

# Play a game against the computer
function play_game(session)
  net = AlphaZero.Network.copy(session.env.bestnn, on_gpu=true, test_mode=true)
  mcts = MCTS.Env{Game}(net, nworkers=64)
  GI.interactive!(Game(), MCTS.AI(mcts, timeout=5.0), GI.Human())
end

play_game(session)
