#####
##### High level training procedure
#####

"""
    Env

Type for an AlphZero environment.

The environment features the current neural network, the best neural network
seen so far that is used for data generation, a memory buffer
and an iteration counter.

# Constructor

    Env(game_spec, params, curnn, bestnn=copy(curnn), experience=[], itc=0)

Construct a new AlphaZero environment:
- `game_spec` specified the game being played
- `params` has type [`Params`](@ref)
- `curnn` is the current neural network and has type [`AbstractNetwork`](@ref)
- `bestnn` is the best neural network so far, which is used for data generation
- `experience` is the initial content of the memory buffer
   as a vector of [`TrainingSample`](@ref)
- `itc` is the value of the iteration counter (0 at the start of training)
"""
mutable struct Env{GameSpec, Network, State}
  gspec  :: GameSpec
  params :: Params
  curnn  :: Network
  bestnn :: Network
  memory :: MemoryBuffer{GameSpec, State}
  itc    :: Int
  function Env(
      gspec::AbstractGameSpec,
      params, curnn, bestnn=copy(curnn), experience=[], itc=0)
    msize = max(params.mem_buffer_size[itc], length(experience))
    memory = MemoryBuffer(gspec, msize, experience)
    return new{typeof(gspec), typeof(curnn), GI.state_type(gspec)}(
      gspec, params, curnn, bestnn, memory, itc)
  end
end

#####
##### Training handlers
#####

"""
    Handlers

Namespace for the callback functions that are used during training.
This enables logging, saving and plotting to be implemented separately.
An example handler object is `Session`.

All callback functions take a handler object `h` as their first argument
and sometimes a second argment `r` that consists in a report.

| Callback                    | Comment                                        |
|:----------------------------|:-----------------------------------------------|
| `iteration_started(h)`      | called at the beggining of an iteration        |
| `self_play_started(h)`      | called once per iter before self play starts   |
| `game_played(h)`            | called after each game of self play            |
| `self_play_finished(h, r)`  | sends report: [`Report.SelfPlay`](@ref)        |
| `memory_analyzed(h, r)`     | sends report: [`Report.Memory`](@ref)          |
| `learning_started(h)`       | called at the beginning of the learning phase  |
| `updates_started(h, r)`     | sends report: [`Report.LearningStatus`](@ref)  |
| `updates_finished(h, r)`    | sends report: [`Report.LearningStatus`](@ref)  |
| `checkpoint_started(h)`     | called before a checkpoint evaluation starts   |
| `checkpoint_game_played(h)` | called after each arena game                   |
| `checkpoint_finished(h, r)` | sends report: [`Report.Checkpoint`](@ref)      |
| `learning_finished(h, r)`   | sends report: [`Report.Learning`](@ref)        |
| `iteration_finished(h, r)`  | sends report: [`Report.Iteration`](@ref)       |
| `training_finished(h)`      | called once at the end of training             |
"""
module Handlers

  import ..Report

  function iteration_started(h)      return end
  function self_play_started(h)      return end
  function game_played(h)            return end
  function self_play_finished(h, r)  return end
  function memory_analyzed(h, r)     return end
  function learning_started(h)       return end
  function updates_started(h)        return end
  function updates_finished(h, r)    return end
  function checkpoint_started(h)     return end
  function checkpoint_game_played(h) return end
  function checkpoint_finished(h, r) return end
  function learning_finished(h, r)   return end
  function iteration_finished(h, r)  return end
  function training_finished(h)      return end

end

#####
##### Public utilities
#####

"""
    get_experience(env::Env)

Return the content of the agent's memory as a vector of [`TrainingSample`](@ref).
"""
get_experience(env::Env) = get_experience(env.memory)

"""
    initial_report(env::Env)

Return a report summarizing the configuration of agent before training starts,
as an object of type [`Report.Initial`](@ref).
"""
function initial_report(env::Env)
  num_network_parameters = Network.num_parameters(env.curnn)
  num_reg_params = Network.num_regularized_parameters(env.curnn)
  player = MctsPlayer(env.gspec, env.curnn, env.params.self_play.mcts)
  mcts_footprint_per_node = MCTS.memory_footprint_per_node(env.gspec)
  errs, warns = check_params(env.gspec, env.params)
  return Report.Initial(
    num_network_parameters, num_reg_params, mcts_footprint_per_node, errs, warns)
end

#####
##### Evaluating networks
#####

# Have a "contender" network play against a "baseline" network (params::ArenaParams)
# Return (rewards vector, redundancy)
# Version for two-player games
function pit_networks(gspec, contender, baseline, params, handler)
  make_oracles() = (
    Network.copy(contender, on_gpu=params.sim.use_gpu, test_mode=true),
    Network.copy(baseline, on_gpu=params.sim.use_gpu, test_mode=true))
  simulator = Simulator(make_oracles, record_trace) do oracles
    white = MctsPlayer(gspec, oracles[1], params.mcts)
    black = MctsPlayer(gspec, oracles[2], params.mcts)
    return TwoPlayers(white, black)
  end
  samples = simulate(
    simulator, gspec, params.sim,
    game_simulated=(() -> Handlers.checkpoint_game_played(handler)))
  return rewards_and_redundancy(samples, gamma=params.mcts.gamma)
end

# Evaluate a single neural network for a one-player game (params::ArenaParams)
function evaluate_network(gspec, net, params, handler)
  make_oracles() = Network.copy(net, on_gpu=params.sim.use_gpu, test_mode=true)
  simulator = Simulator(make_oracles, record_trace) do oracle
    MctsPlayer(gspec, oracle, params.mcts)
  end
  samples = simulate(
    simulator, gspec, params.sim,
    game_simulated=(() -> Handlers.checkpoint_game_played(handler)))
  return rewards_and_redundancy(samples, gamma=params.mcts.gamma)
end

# Compare two versions of a neural network (params::ArenaParams)
# Works for both two-player and single-player games
function compare_networks(gspec, contender, baseline, params, handler)
  legend = "Most recent NN versus best NN so far"
  if GI.two_players(gspec)
    (rewards_c, red), t =
      @timed pit_networks(gspec, contender, baseline, params, handler)
    avgr = mean(rewards_c)
    rewards_b = nothing
  else
    (rewards_c, red_c), tc = @timed evaluate_network(gspec, contender, params, handler)
    (rewards_b, red_b), tb = @timed evaluate_network(gspec, baseline, params, handler)
    avgr = mean(rewards_c) - mean(rewards_b)
    red = mean([red_c, red_b])
    t = tc + tb
  end
  return Report.Evaluation(legend, avgr, red, rewards_c, rewards_b, t)
end

#####
##### Training Loop
#####

function resize_memory!(env::Env, n)
  exp = get_experience(env.memory)
  env.memory = MemoryBuffer(env.gspec, n, exp)
  return
end

function dummy_learning_report()
  eps = 1e-16
  dummy_loss = Report.Loss(0, 0, 0, 0, 0)
  dummy_status = Report.LearningStatus(dummy_loss, 0, 0)
  return Report.Learning(eps, eps, eps, eps, dummy_status, [], [], false)
end

function learning_step!(env::Env, handler)
  ap = env.params.arena
  lp = env.params.learning
  checkpoints = Report.Checkpoint[]
  losses = Float32[]
  tloss, teval, ttrain = 0., 0., 0.
  experience = get_experience(env.memory)
  if env.params.use_symmetries
    experience = augment_with_symmetries(env.gspec, experience)
  end
  if isempty(experience)
    # Skipping the learning phase
    return dummy_learning_report()
  end
  trainer, tconvert = @timed Trainer(env.gspec, env.curnn, experience, lp)
  init_status = learning_status(trainer)
  status = init_status
  Handlers.learning_started(handler)
  # Compute the number of batches between each checkpoint
  nbatches = lp.max_batches_per_checkpoint
  if !iszero(lp.min_checkpoints_per_epoch)
    ntotal = num_batches_total(trainer)
    nbatches = min(nbatches, ntotal ÷ lp.min_checkpoints_per_epoch)
  end
  # Loop state variables
  best_evalr = isnothing(ap) ? nothing : ap.update_threshold
  nn_replaced = false

  for k in 1:lp.num_checkpoints
    # Execute a series of batch updates
    Handlers.updates_started(handler, status)
    dlosses, dttrain = @timed batch_updates!(trainer, nbatches)
    status, dtloss = @timed learning_status(trainer)
    Handlers.updates_finished(handler, status)
    tloss += dtloss
    ttrain += dttrain
    append!(losses, dlosses)
    # Run a checkpoint evaluation if the arena parameter is provided
    if isnothing(ap)
      env.curnn = get_trained_network(trainer)
      env.bestnn = copy(env.curnn)
      nn_replaced = true
    else
      Handlers.checkpoint_started(handler)
      env.curnn = get_trained_network(trainer)
      eval_report =
        compare_networks(env.gspec, env.curnn, env.bestnn, ap, handler)
      teval += eval_report.time
      # If eval is good enough, replace network
      success = (eval_report.avgr >= best_evalr)
      if success
        nn_replaced = true
        env.bestnn = copy(env.curnn)
        best_evalr = eval_report.avgr
      end
      checkpoint_report =
        Report.Checkpoint(k * nbatches, eval_report, status, success)
      push!(checkpoints, checkpoint_report)
      Handlers.checkpoint_finished(handler, checkpoint_report)
    end
  end
  report = Report.Learning(
    tconvert, tloss, ttrain, teval,
    init_status, losses, checkpoints, nn_replaced)
  Handlers.learning_finished(handler, report)
  return report
end

function simple_memory_stats(env::Env)
  mem = get_experience(env)
  nsamples = length(mem)
  ndistinct = length(merge_by_state(mem))
  return nsamples, ndistinct
end

# To be given as an argument to `Simulator`
function self_play_measurements(trace, _, player)
  mem = MCTS.approximate_memory_footprint(player.mcts)
  edepth = MCTS.average_exploration_depth(player.mcts)
  return (trace=trace, mem=mem, edepth=edepth)
end

function self_play_step!(env::Env, handler)
  params = env.params.self_play
  Handlers.self_play_started(handler)
  make_oracle() =
    Network.copy(env.bestnn, on_gpu=params.sim.use_gpu, test_mode=true)
  simulator = Simulator(make_oracle, self_play_measurements) do oracle
    return MctsPlayer(env.gspec, oracle, params.mcts)
  end
  # Run the simulations
  results, elapsed = @timed simulate_distributed(
    simulator, env.gspec, params.sim,
    game_simulated=()->Handlers.game_played(handler))
  # Add the collected samples in memory
  new_batch!(env.memory)
  for x in results
    push_trace!(env.memory, x.trace, params.mcts.gamma)
  end
  speed = cur_batch_size(env.memory) / elapsed
  edepth = mean([x.edepth for x in results])
  mem_footprint = maximum([x.mem for x in results])
  memsize, memdistinct = simple_memory_stats(env)
  report = Report.SelfPlay(
    speed, edepth, mem_footprint, memsize, memdistinct)
  Handlers.self_play_finished(handler, report)
  return report
end

function memory_report(env::Env, handler)
  if isnothing(env.params.memory_analysis)
    return nothing
  else
    report = memory_report(
      env.memory, env.curnn, env.params.learning, env.params.memory_analysis)
    Handlers.memory_analyzed(handler, report)
    return report
  end
end

"""
    train!(env::Env, handler=nothing)

Start or resume the training of an AlphaZero agent.

A `handler` object can be passed that implements a subset of the callback
functions defined in [`Handlers`](@ref).
"""
function train!(env::Env, handler=nothing)
  while env.itc < env.params.num_iters
    Handlers.iteration_started(handler)
    resize_memory!(env, env.params.mem_buffer_size[env.itc])
    sprep, spperfs = Report.@timed self_play_step!(env, handler)
    mrep, mperfs = Report.@timed memory_report(env, handler)
    lrep, lperfs = Report.@timed learning_step!(env, handler)
    rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(handler, rep)
  end
  Handlers.training_finished(handler)
end

#####
##### AlphaZero player
#####

function guess_mcts_arena_params(env::Env)
  p = env.params
  return isnothing(p.arena) ? p.self_play.mcts : p.arena.mcts
end

function guess_use_gpu(env::Env)
  p = env.params
  return isnothing(p.arena) ? p.self_play.sim.use_gpu : p.arena.sim.use_gpu
end

"""
    AlphaZeroPlayer(::Env; [timeout, mcts_params, use_gpu])

Create an AlphaZero player from the current [training environment](@ref environment).

Note that the returned player may be slow as it does not batch MCTS requests.
"""
function AlphaZeroPlayer(env::Env; timeout=2.0, mcts_params=nothing, use_gpu=nothing)
  isnothing(mcts_params) && (mcts_params = guess_mcts_arena_params(env))
  isnothing(use_gpu) && (use_gpu = guess_use_gpu(env))
  net = Network.copy(env.bestnn, on_gpu=use_gpu, test_mode=true)
  return MctsPlayer(env.gspec, net, mcts_params, timeout=timeout)
end