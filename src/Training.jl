#####
##### High level training procedure
#####

mutable struct Env{Game, Network, Board}
  params :: Params
  bestnn :: Network
  memory :: MemoryBuffer{Board}
  itc    :: Int
  randnn :: Bool # true if `bestnn` has random weights
  function Env{Game}(params, network, experience=[], itc=0) where Game
    Board = GI.Board(Game)
    msize = max(get(params.mem_buffer_size, itc), length(experience))
    memory = MemoryBuffer{Board}(msize, experience)
    randnn = itc == 0
    return new{Game, typeof(network), Board}(
      params, network, memory, itc, randnn)
  end
end

#####
##### Training handlers
#####

module Handlers

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
##### Training loop
#####

get_experience(env::Env) = get(env.memory)

function resize_memory!(env::Env{G,N,B}, n) where {G,N,B}
  exp = get(env.memory)
  env.memory = MemoryBuffer{B}(n, exp)
end

function initial_report(env::Env)
  num_network_parameters = Network.num_parameters(env.bestnn)
  num_reg_params = Network.num_regularized_parameters(env.bestnn)
  player = MctsPlayer(env.bestnn, env.params.self_play.mcts)
  mcts_footprint_per_node = MCTS.memory_footprint_per_node(player.mcts)
  return Report.Initial(
    num_network_parameters, num_reg_params, mcts_footprint_per_node)
end

function evaluate_network(baseline, contender, params, handler)
  baseline = MctsPlayer(baseline, params.mcts)
  contender = MctsPlayer(contender, params.mcts)
  ngames = params.num_games
  reset_period = params.reset_mcts_every
  return pit(baseline, contender, ngames, reset_period) do i, z
    Handlers.checkpoint_game_played(handler)
  end
end

function learning!(env::Env{G}, handler) where G
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
    ttrain += @elapsed training_epoch!(trainer)
    status, dtloss = @timed learning_status(trainer)
    tloss += dtloss
    epoch_report = Report.Epoch(status)
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
      if i % params.reset_mcts_every == 0 || i == params.num_games
        mem_footprint = max(mem_footprint,
          MCTS.approximate_memory_footprint(player.mcts))
        MCTS.reset!(player.mcts)
      end
    end
  end
  MCTS.memory_footprint(player.mcts)
  inference_tr = MCTS.inference_time_ratio(player.mcts)
  speed = last_batch_size(env.memory) / elapsed
  expdepth = MCTS.average_exploration_depth(player.mcts)
  report = Report.SelfPlay(inference_tr, speed, expdepth, mem_footprint)
  Handlers.self_play_finished(handler, report)
  return report
end

function memory_report(env::Env{G}, handler) where G
  nstages = env.params.num_game_stages
  report = memory_report(env.memory, env.bestnn, env.params.learning, nstages)
  Handlers.memory_analyzed(handler, report)
  return report
end

function train!(env::Env{G}, handler=nothing) where G
  while env.itc < env.params.num_iters
    Handlers.iteration_started(handler)
    resize_memory!(env, get(env.params.mem_buffer_size, env.itc))
    sprep, sptime = @timed self_play!(env, handler)
    mrep, mtime = @timed memory_report(env, handler)
    lrep, ltime = @timed learning!(env, handler)
    rep = Report.Iteration(sptime, mtime, ltime, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(handler, rep)
  end
  Handlers.training_finished(handler)
end
