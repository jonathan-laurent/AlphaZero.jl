#####
##### Dummy runs are used to ensure the absence of runtime errors in the code
##### before launching a training session
#####

using Setfield

dummy_run_mcts(p::MctsParams) = @set p.num_iters_per_turn = 2

dummy_run_player(p::Benchmark.Player) = p
dummy_run_player(p::Benchmark.Full) = @set p.params = dummy_run_mcts(p.params)
dummy_run_player(p::Benchmark.MctsRollouts) = @set p.params = dummy_run_mcts(p.params)

# Returned modified parameters where all num_games fields are set to 1.
# The number of iterations is set to 2.
function dummy_run_params(params)
  params = @set params.self_play.num_games = 1
  params = @set params.self_play.num_workers = 1
  params = @set params.self_play.mcts = dummy_run_mcts(params.self_play.mcts)
  if !isnothing(params.arena)
    params = @set params.arena.num_games = 1
    params = @set params.arena.mcts = dummy_run_mcts(params.arena.mcts)
  end
  params = @set params.learning.max_batches_per_checkpoint = 2
  params = @set params.learning.num_checkpoints = min(params.learning.num_checkpoints, 2)
  params = @set params.num_iters = 2
  return params
end

function dummy_run_duel(d)
  d = @set d.num_games = 1
  d = @set d.num_workers = 1
  d = @set d.player = dummy_run_player(d.player)
  d = @set d.baseline = dummy_run_player(d.baseline)
  return d
end

dummy_run_benchmark(benchmark) = map(dummy_run_duel, benchmark)

function dummy_run_experiment(e::Experiment)
  params = dummy_run_params(e.params)
  benchmark = dummy_run_benchmark(e.benchmark)
  return Experiment(e.name, e.gspec, params, e.mknet, e.netparams, benchmark=benchmark)
end

function dummy_run(experiment::Experiment; session_dir=nothing, nostdout=true)
  experiment = dummy_run_experiment(experiment)
  isnothing(session_dir) && (session_dir = "sessions/test-$(experiment.name)")
  session = Session(experiment, nostdout=nostdout, dir=session_dir)
  resume!(session)
  return true
end
