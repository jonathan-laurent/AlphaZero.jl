#####
##### High level training procedure
#####

"""
    Env{Game, Network, Board}

Type for an AlphZero environment.

The environment features the current best neural network, a memory buffer
and an iteration counter.

# Constructor

    Env{Game}(params, network, experience=[], itc=0)

Construct a new AlphaZero environment.
- `Game` is the type of the game being played
- `params` has type [`Params`](@ref)
- `network` is the initial neural network and has type [`AbstractNetwork`](@ref)
- `experience` is the initial content of the memory buffer
   as a vector of [`TrainingSample`](@ref)
- `itc` is the value of the iteration counter (0 at the start of training)
"""
mutable struct Env{Game, Network, Board}
  params :: Params
  bestnn :: Network
  memory :: MemoryBuffer{Board}
  itc    :: Int
  randnn :: Bool # true if `bestnn` has random weights
  function Env{Game}(params, network, experience=[], itc=0) where Game
    Board = GI.Board(Game)
    msize = max(params.mem_buffer_size[itc], length(experience))
    memory = MemoryBuffer{Board}(msize, experience)
    randnn = itc == 0
    return new{Game, typeof(network), Board}(
      params, network, memory, itc, randnn)
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
| `learning_started(h, r)`    | sends report: [`Report.LearningStatus`](@ref)  |
| `learning_epoch(h, r)`      | sends report: [`Report.Epoch`](@ref)           |
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
  function learning_epoch(h, r)      return end
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
get_experience(env::Env) = get(env.memory)

"""
    initial_report(env::Env)

Return a report summarizing the configuration of agent before training starts,
as an object of type [`Report.Initial`](@ref).
"""
function initial_report(env::Env)
  num_network_parameters = Network.num_parameters(env.bestnn)
  num_reg_params = Network.num_regularized_parameters(env.bestnn)
  player = MctsPlayer(env.bestnn, env.params.self_play.mcts)
  mcts_footprint_per_node = MCTS.memory_footprint_per_node(player.mcts)
  return Report.Initial(
    num_network_parameters, num_reg_params, mcts_footprint_per_node)
end

#####
##### Training loop
#####

function resize_memory!(env::Env{G,N,B}, n) where {G,N,B}
  exp = get(env.memory)
  env.memory = MemoryBuffer{B}(n, exp)
end

function evaluate_network(baseline, contender, params, handler)
  baseline = MctsPlayer(baseline, params.mcts)
  contender = MctsPlayer(contender, params.mcts)
  ngames = params.num_games
  rp = params.reset_mcts_every
  return pit(contender, baseline, ngames; reset_every=rp) do i, z
    Handlers.checkpoint_game_played(handler)
  end
end

function learning!(env::Env, handler)
  # Initialize the training process
  ap = env.params.arena
  lp = env.params.learning
  epochs = Report.Epoch[]
  checkpoints = Report.Checkpoint[]
  tloss, teval, ttrain = 0., 0., 0.
  trainer, tconvert = @timed Trainer(env.bestnn, get(env.memory), lp)
  init_status = learning_status(trainer)
  Handlers.learning_started(handler, init_status)
  # Loop state variables
  best_evalz = ap.update_threshold
  nn_replaced = false
  # Loop over epochs
  for k in 1:maximum(lp.checkpoints)
    # Execute learning epoch
    losses, dttrain = @timed training_epoch!(trainer)
    status, dtloss = @timed learning_status(trainer)
    tloss += dtloss
    ttrain += dttrain
    epoch_report = Report.Epoch(status, losses)
    push!(epochs, epoch_report)
    Handlers.learning_epoch(handler, epoch_report)
    # Make checkpoints at fixed times
    if k ∈ lp.checkpoints
      Handlers.checkpoint_started(handler)
      cur_nn = get_trained_network(trainer)
      evalz, dteval = @timed evaluate_network(env.bestnn, cur_nn, ap, handler)
      teval += dteval
      # If eval is good enough, replace network
      success = evalz >= best_evalz
      if success
        nn_replaced = true
        env.bestnn = cur_nn
        env.randnn = false
        best_evalz = evalz
      end
      checkpoint_report = Report.Checkpoint(k, evalz, success)
      push!(checkpoints, checkpoint_report)
      Handlers.checkpoint_finished(handler, checkpoint_report)
    end
  end
  report = Report.Learning(
    tconvert, tloss, ttrain, teval,
    init_status, epochs, checkpoints, nn_replaced)
  Handlers.learning_finished(handler, report)
  return report
end

function simple_memory_stats(env)
  mem = get_experience(env)
  nsamples = length(mem)
  ndistinct = length(merge_by_board(mem))
  return nsamples, ndistinct
end

function self_play!(env::Env{G}, handler) where G
  params = env.params.self_play
  player = env.randnn ?
    RandomMctsPlayer(G, params.mcts) :
    MctsPlayer(env.bestnn, params.mcts)
  new_batch!(env.memory)
  Handlers.self_play_started(handler)
  mem_footprint = 0
  elapsed = @elapsed begin
    for i in 1:params.num_games
      self_play!(player, env.memory)
      Handlers.game_played(handler)
      reset_every = params.reset_mcts_every
      if (!isnothing(reset_every) && i % reset_every == 0) ||
          i == params.num_games
        mem_footprint = max(mem_footprint,
          MCTS.approximate_memory_footprint(player.mcts))
        MCTS.reset!(player.mcts)
      end
      if !isnothing(params.gc_every) && i % params.gc_every == 0
        Network.gc(env.bestnn)
      end
    end
  end
  MCTS.memory_footprint(player.mcts)
  inference_tr = MCTS.inference_time_ratio(player.mcts)
  speed = last_batch_size(env.memory) / elapsed
  expdepth = MCTS.average_exploration_depth(player.mcts)
  memsize, memdistinct = simple_memory_stats(env)
  report = Report.SelfPlay(
    inference_tr, speed, expdepth, mem_footprint, memsize, memdistinct)
  Handlers.self_play_finished(handler, report)
  return report
end

function memory_report(env::Env, handler)
  if isnothing(env.params.memory_analysis)
    return nothing
  else
    report = memory_report(
      env.memory, env.bestnn, env.params.learning, env.params.memory_analysis)
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
    sprep, spperfs = Report.@timed self_play!(env, handler)
    mrep, mperfs = Report.@timed memory_report(env, handler)
    lrep, lperfs = Report.@timed learning!(env, handler)
    rep = Report.Iteration(spperfs, mperfs, lperfs, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(handler, rep)
  end
  Handlers.training_finished(handler)
end
