#####
##### High level training procedure
#####

mutable struct Env{Game, Network, Board, Mcts}
  params :: Params
  bestnn :: Network
  memory :: MemoryBuffer{Board}
  mcts   :: Mcts
  itc    :: Int
  function Env{Game}(params, network, experience=[], itc=0) where Game
    Board = GI.Board(Game)
    memory = MemoryBuffer{Board}(params.mem_buffer_size, experience)
    mcts = MCTS.Env{Game}(network, params.self_play.mcts.cpuct)
    env = new{Game, typeof(network), Board, typeof(mcts)}(
      params, network, memory, mcts, itc)
    update_network!(env, network)
    return env
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
  function learning_checkpoint(h, r) return end
  function learning_finished(h, r)   return end
  function iteration_finished(h, r)  return end
  function training_finished(h)      return end

end

import .Handlers

#####
##### Training loop
#####

function update_network!(env::Env, net)
  env.bestnn = net
  MCTS.reset!(env.mcts, net)
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
  k = 1 # epoch number
  best_eval_score = ap.update_threshold
  best_nn = env.bestnn
  nn_replaced = false
  last_loss = init_status.loss.L
  stable_loss = false
  # Loop over epochs
  while !stable_loss && k <= lp.max_num_epochs &&
        !(lp.stop_after_first_winner && nn_replaced)
    ttrain += @elapsed training_epoch!(trainer)
    status, dtloss = @timed learning_status(trainer)
    tloss += dtloss
    stable_loss = (last_loss - status.loss.L < lp.stop_loss_eps)
    last_loss = status.loss.L
    epoch_report = Report.Epoch(status, stable_loss)
    push!(epochs, epoch_report)
    # Decide whether or not to make a checkpoint
    if stable_loss || k % lp.epochs_per_checkpoint == 0
      cur_nn = get_trained_network(trainer)
      eval_reward, dteval = @timed evaluate_network(env.bestnn, cur_nn, ap)
      teval += dteval
      # If eval is good enough, replace network
      success = eval_reward >= best_eval_score
      if success
        nn_replaced = true
        best_nn = cur_nn
        best_eval_score = eval_reward
      end
      checkpoint_report = Report.Checkpoint(k, eval_reward, success)
      push!(checkpoints, checkpoint_report)
      Handlers.learning_checkpoint(handler, checkpoint_report)
    end
    Handlers.learning_epoch(handler, epoch_report)
    k += 1
  end
  nn_replaced && update_network!(env, best_nn)
  report = Report.Learning(
    tconvert, tloss, ttrain, teval,
    init_status, epochs, checkpoints, nn_replaced)
  Handlers.learning_finished(handler, report)
  return report
end

function self_play!(env::Env{G}, handler) where G
  params = env.params.self_play
  player = MctsPlayer(env.bestnn, params.mcts)
  new_batch!(env.memory)
  Handlers.self_play_started(handler)
  elapsed = @elapsed begin
    for i in 1:params.num_games
      self_play!(player, env.memory)
      Handlers.game_played(handler)
    end
  end
  inference_tr = MCTS.inference_time_ratio(player.mcts)
  speed = last_batch_size(env.memory) / elapsed
  report = Report.SelfPlay(inference_tr, speed)
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
    sprep, sptime = @timed self_play!(env, handler)
    mrep, mtime = @timed memory_report(env, handler)
    lrep, ltime = @timed learning!(env, handler)
    rep = Report.Iteration(sptime, mtime, ltime, sprep, mrep, lrep)
    env.itc += 1
    Handlers.iteration_finished(handler, rep)
  end
  Handlers.training_finished(handler)
end
