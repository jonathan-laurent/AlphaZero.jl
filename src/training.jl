#####
##### High level training procedure
#####

"""
    Env{Game, Network, Board}

Type for an AlphZero environment.

The environment features the current neural network, the best neural network
seen so far that is used for data generation, a memory buffer
and an iteration counter.

# Constructor

    Env{Game}(params, curnn, bestnn=copy(curnn), experience=[], itc=0)

Construct a new AlphaZero environment.
- `Game` is the type of the game being played
- `params` has type [`Params`](@ref)
- `curnn` is the current neural network and has type [`AbstractNetwork`](@ref)
- `bestnn` is the best neural network so far, which is used for data generation
- `experience` is the initial content of the memory buffer
   as a vector of [`TrainingSample`](@ref)
- `itc` is the value of the iteration counter (0 at the start of training)
"""
mutable struct Env{Game, Network, Board}
  params :: Params
  curnn  :: Network
  bestnn :: Network
  memory :: MemoryBuffer{Board}
  itc    :: Int
  function Env{Game}(
      params, curnn, bestnn=copy(curnn), experience=[], itc=0) where Game
    Board = GI.State(Game)
    msize = max(params.mem_buffer_size[itc], length(experience))
    memory = MemoryBuffer{Board}(msize, experience)
    return new{Game, typeof(curnn), Board}(
      params, curnn, bestnn, memory, itc)
  end
end

GameType(env::Env{Game}) where Game = Game

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
| `learning_started(h, r)`    | sends report: [`Report.LearningStatus`](@ref)  |
| `updates_started(h)`        | called before each series of batch updates     |
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
  function learning_started(h, r)    return end
  function updates_started(h)        return end
  function updates_finished(h, r)    return end
  function checkpoint_started(h)     return end
  function checkpoint_game_played(h) return end
  function checkpoint_finished(h, r) return end
  function learning_finished(h, r)   return end
  function iteration_finished(h, r)  return end
  function training_finished(h)      return end

end

import .Handlers

#####
##### Public utilities
#####

"""
    get_experience(env::Env)

Return the content of the agent's memory as a
vector of [`TrainingSample`](@ref).
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
  player = MctsPlayer(env.curnn, env.params.self_play.mcts)
  mcts_footprint_per_node = MCTS.memory_footprint_per_node(player.mcts)
  return Report.Initial(
    num_network_parameters, num_reg_params, mcts_footprint_per_node)
end

#####
##### Training loop
#####

function resize_memory!(env::Env{G,N,B}, n) where {G,N,B}
  exp = get_experience(env.memory)
  env.memory = MemoryBuffer{B}(n, exp)
  return
end

function evaluate_network(contender, baseline, params, handler)
  contender = MctsPlayer(contender, params.arena.mcts)
  baseline = MctsPlayer(baseline, params.arena.mcts)
  ngames = params.arena.num_games
  states = []
  gamma = params.self_play.mcts.gamma
  avgz = pit(contender, baseline, ngames; gamma=gamma,
      reset_every=params.arena.reset_mcts_every,
      flip_probability=params.arena.flip_probability) do i, z, t
    Handlers.checkpoint_game_played(handler)
    append!(states, t.states)
  end
  redundancy = compute_redundancy(GameType(contender), states)
  return avgz, redundancy
end

function learning_step!(env::Env, handler)
  ap = env.params.arena
  lp = env.params.learning
  checkpoints = Report.Checkpoint[]
  losses = Float32[]
  tloss, teval, ttrain = 0., 0., 0.
  experience = get_experience(env.memory)
  if env.params.use_symmetries
    experience = augment_with_symmetries(GameType(env), experience)
  end
  trainer, tconvert = @timed Trainer(env.curnn, experience, lp)
  init_status = learning_status(trainer)
  Handlers.learning_started(handler, init_status)
  # Compute the number of batches between each checkpoint
  nbatches = lp.max_batches_per_checkpoint
  if !iszero(lp.min_checkpoints_per_epoch)
    ntotal = num_batches_total(trainer)
    nbatches = min(nbatches, ntotal ÷ lp.min_checkpoints_per_epoch)
  end
  # Loop state variables
  best_evalz = ap.update_threshold
  nn_replaced = false

  for k in 1:lp.num_checkpoints
    # Execute a series of batch updates
    Handlers.updates_started(handler)
    dlosses, dttrain = @timed batch_updates!(trainer, nbatches)
    status, dtloss = @timed learning_status(trainer)
    Handlers.updates_finished(handler, status)
    tloss += dtloss
    ttrain += dttrain
    append!(losses, dlosses)
    # Run a checkpoint evaluation
    Handlers.checkpoint_started(handler)
    env.curnn = get_trained_network(trainer)
    (evalz, redundancy), dteval =
      @timed evaluate_network(env.curnn, env.bestnn, env.params, handler)
    teval += dteval
    # If eval is good enough, replace network
    success = (evalz >= best_evalz)
    if success
      nn_replaced = true
      env.bestnn = copy(env.curnn)
      best_evalz = evalz
    end
    checkpoint_report = Report.Checkpoint(
      k * nbatches, status, evalz, redundancy, success)
    push!(checkpoints, checkpoint_report)
    Handlers.checkpoint_finished(handler, checkpoint_report)
  end

  report = Report.Learning(
    tconvert, tloss, ttrain, teval,
    init_status, losses, checkpoints, nn_replaced)
  Handlers.learning_finished(handler, report)
  return report
end

function simple_memory_stats(env)
  mem = get_experience(env)
  nsamples = length(mem)
  ndistinct = length(merge_by_state(mem))
  return nsamples, ndistinct
end

# Run `num_sims` game simulations and
# return a (traces, mem, expdepth) named-tuple.
function self_play_worker(oracle, params, lock, handler, num_sims)
  player = MctsPlayer(oracle, params.mcts)
  res = map(1:num_sims) do i
    trace = play_game(player)
    Base.lock(lock)
    Handlers.game_played(handler)
    unlock(lock)
    mem = MCTS.approximate_memory_footprint(player.mcts)
    reset_every = params.reset_mcts_every
    if !isnothing(reset_every) && i % reset_every == 0
      MCTS.reset!(player.mcts)
    end
    if !isnothing(params.gc_every) && i % params.gc_every == 0
      Network.gc(env.bestnn)
    end
    return (trace=trace, mem=mem)
  end
  mem = maximum([r.mem for r in res])
  expdepth = MCTS.average_exploration_depth(player.mcts)
  return (traces=[r.trace for r in res], mem=mem, expdepth=expdepth)
end

function self_play_step!(env::Env{G}, handler) where G
  params = env.params.self_play
  Handlers.self_play_started(handler)
  network = Network.copy(env.bestnn, on_gpu=params.use_gpu, test_mode=true)
  lock = ReentrantLock()
  reqc = Batchifier.launch_server(params.num_workers) do state
    Network.evaluate_batch(network, state)
  end
  # For each worker
  @assert params.num_workers <= params.num_games
  res, elapsed = @timed Util.threads_pmap(1:params.num_workers) do _
    oracle = Batchifier.BatchedOracle{G}(reqc)
    num_sims = params.num_games ÷ params.num_workers
    res = self_play_worker(oracle, params, lock, handler, num_sims)
    Batchifier.done!(oracle)
    return res
  end
  traces = [t for r in res for t in r.traces]
  new_batch!(env.memory)
  for trace in traces
    push_game!(env.memory, trace, params.mcts.gamma)
  end
  speed = cur_batch_size(env.memory) / elapsed
  expdepth = mean([r.expdepth for r in res])
  mem_footprint = sum([r.mem for r in res])
  memsize, memdistinct = simple_memory_stats(env)
  report = Report.SelfPlay(
    speed, expdepth, mem_footprint, memsize, memdistinct)
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
