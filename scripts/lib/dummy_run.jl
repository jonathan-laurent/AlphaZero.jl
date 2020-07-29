#####
##### Dummy runs are used to ensure the absence of runtime errors in the code
##### before launching a training session
#####

function with_dummy_mcts(mcts::MctsParams)
  return MctsParams(mcts; num_iters_per_turn=2)
end

function with_dummy_mcts(p::SelfPlayParams)
  return SelfPlayParams(p; mcts=with_dummy_mcts(p.mcts))
end

function with_dummy_mcts(p::ArenaParams)
  return ArenaParams(p; mcts=with_dummy_mcts(p.mcts))
end

function with_dummy_mcts(p::Benchmark.Full)
  return Benchmark.Full(with_dummy_mcts(p.params))
end

function with_dummy_mcts(p::Benchmark.MctsRollouts)
  return Benchmark.MctsRollouts(with_dummy_mcts(p.params))
end

with_dummy_mcts(p::Benchmark.Player) = p

with_dummy_mcts(::Nothing) = nothing

# Returned modified parameters where all num_games fields are set to 1.
# The number of iterations is set to 2.
function dummy_run_params(params, benchmark)
  self_play = SelfPlayParams(
    with_dummy_mcts(params.self_play),
    num_games=1, num_workers=1)
  arena = nothing
  if !isnothing(params.arena)
    arena = ArenaParams(with_dummy_mcts(params.arena), num_games=1)
  end
  learning = LearningParams(
    params.learning,
    max_batches_per_checkpoint=2,
    num_checkpoints=min(params.learning.num_checkpoints, 2))
  params = Params(
    params, num_iters=2,
    self_play=self_play, arena=arena, learning=learning)
  benchmark = [
    Benchmark.Duel(
      with_dummy_mcts(d.player), with_dummy_mcts(d.baseline),
      num_games=1, num_workers=1, color_policy=d.color_policy)
    for d in benchmark ]
  return params, benchmark
end

function dummy_run(GameModule; session_dir=nothing, nostdout=true)
  Game = GameModule.Game
  Training = GameModule.Training
  Net = Training.Network
  netparams = Training.netparams
  params, benchmark = dummy_run_params(Training.params, Training.benchmark)
  session = Session(Game, Net{Game}, params, netparams,
    benchmark=benchmark, nostdout=nostdout, dir=session_dir)
  resume!(session)
  return true
end
