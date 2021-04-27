#####
##### Profiling MCTS
#####

using AlphaZero
using Base: @timed

# using Revise; Revise.includet("scripts/profile/mcts.jl")
# profile_mcts(nrep=64, nworkers=64, batched=true)

# Cost of simulation on one thread: 11μs
# Cost of simulation on all threads: 3μs
# To be compared to the cost of inference: 15μs to 30μs
# Also, by comparing `profile_mcts` with `batched=true` and `batched=false`, we conclude
# that the overhead of spawning tasks is not too big.

"""
Return the search time per inference request sent in μs/request.

!!! note
    For both the GPU and CPU to be maximally utilized, this number must be comparable to
    the result of `profile_inference`.
"""
function profile_mcts(
  exp::Experiment = Examples.experiments["connect-four"];
  batched=false,
  nrep=30,
  nworkers=1)

  oracle = MCTS.RandomOracle(exp.gspec)
  if batched
    spawn_oracle, done! =
      AlphaZero.batchify_oracles(
        oracle;
        num_workers=nworkers,
        batch_size=nworkers,
        fill_batches=false)
    oracles = [spawn_oracle() for _ in 1:nworkers]
  else
    oracles = [oracle for _ in 1:nworkers]
    done!() = nothing
  end

  players = [
    MctsPlayer(exp.gspec, oracle, exp.params.self_play.mcts)
    for oracle in oracles ]

  think(players[1], GI.init(exp.gspec)) # compile everything
  AlphaZero.reset_player!(players[1])

  info = @timed Threads.@threads for player in players
    for i in 1:nrep
      think(player, GI.init(exp.gspec))
    end
    done!()
  end
  traversed = sum(p.mcts.total_simulations for p in players)
  return info.time * 1_000_000 / traversed
end

# profile_mcts(nrep=64, nworkers=1)
# profile_mcts(nrep=64, nworkers=64)
# profile_mcts(nrep=64, nworkers=64, batched=true)