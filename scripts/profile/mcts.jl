#####
##### Profiling MCTS
#####

using AlphaZero
using Base: @timed

# using Revise; Revise.includet("scripts/profile/mcts.jl")
# profile_inference(on_gpu=true, batch_size=128, num_filters=128)

# Takeaway: the cost of simulation is almost zero compared to the cost of evaluating
# the network.

function profile_mcts(
  exp::Experiment = Examples.experiments["connect-four"];
  nrep=100,
  nthreads=128)
  """
  Return the search time per inference request sent in μs.
  Try with many parallel tasks
  """

  oracle = MCTS.RandomOracle(exp.gspec)
  players = [
    MctsPlayer(exp.gspec, oracle, exp.params.self_play.mcts)
    for _ in 1:nthreads ]

  think(players[1], GI.init(exp.gspec)) # compile everything

  info = @timed Threads.@threads for player in players
    game = GI.init(exp.gspec)
    for i in 1:nrep
      think(player, game)
    end
  end
  traversed = sum(p.mcts.total_nodes_traversed for p in players)
  @show traversed
  return info.time * 1_000_000 / traversed
end