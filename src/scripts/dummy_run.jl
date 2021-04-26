#####
##### Dummy runs are used to ensure the absence of runtime errors in the code
##### before launching a training session
#####

using Setfield

dummy_run_mcts(p::MctsParams) = @set p.num_iters_per_turn = 2

dummy_run_player(p::Benchmark.Player) = p
dummy_run_player(p::Benchmark.Full) = @set p.params = dummy_run_mcts(p.params)
dummy_run_player(p::Benchmark.MctsRollouts) = @set p.params = dummy_run_mcts(p.params)

function dummy_run_sim(s::SimParams)
  s = @set s.num_games = 1
  s = @set s.num_workers = 1
  s = @set s.batch_size = 1
  return s
end

# Returned modified parameters where all num_games fields are set to 1.
# The number of iterations is set to 2.
function dummy_run_params(params)
  params = @set params.self_play.sim = dummy_run_sim(params.self_play.sim)
  params = @set params.self_play.mcts = dummy_run_mcts(params.self_play.mcts)
  if !isnothing(params.arena)
    params = @set params.arena.sim = dummy_run_sim(params.arena.sim)
    params = @set params.arena.mcts = dummy_run_mcts(params.arena.mcts)
  end
  params = @set params.learning.max_batches_per_checkpoint = 2
  params = @set params.learning.num_checkpoints = min(params.learning.num_checkpoints, 2)
  params = @set params.num_iters = 2
  return params
end

function dummy_run_evaluation(d::Benchmark.Single)
  d = @set d.sim = dummy_run_sim(d.sim)
  d = @set d.player = dummy_run_player(d.player)
  return d
end

function dummy_run_evaluation(d::Benchmark.Duel)
  d = @set d.sim = dummy_run_sim(d.sim)
  d = @set d.player = dummy_run_player(d.player)
  d = @set d.baseline = dummy_run_player(d.baseline)
  return d
end

dummy_run_benchmark(benchmark) = map(dummy_run_evaluation, benchmark)

function dummy_run_experiment(e::Experiment)
  params = dummy_run_params(e.params)
  benchmark = dummy_run_benchmark(e.benchmark)
  return Experiment(e.name, e.gspec, params, e.mknet, e.netparams, benchmark)
end

"""

    dummy_run(experiment; [dir, nostdout])

Launch a training session where hyperparameters are altered so that training
finishes as quickly as possible.

This is useful to ensure the absence of runtime errors before
a real training session is started.
"""
function dummy_run(experiment::Experiment; dir=nothing, nostdout=false)
  experiment = dummy_run_experiment(experiment)
  isnothing(dir) && (dir = "sessions/dummy-$(experiment.name)")
  rm(dir, force=true, recursive=true)
  session = Session(experiment; dir, nostdout)
  resume!(session)
  return
end
