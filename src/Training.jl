#####
##### High level training procedure
#####

mutable struct Env{Game, Board, Mcts}
  params :: Params
  memory :: MemoryBuffer{Board}
  bestnn :: Oracle{Game}
  mcts   :: Mcts
  logger :: Logger
  function Env{Game}(params) where Game
    Board = GI.Board(Game)
    memory = MemoryBuffer{Board}(params.mem_buffer_size)
    oracle = Oracle{Game}()
    mcts = MCTS.Env{Game}(oracle, params.self_play.cpuct)
    logger = Logger()
    new{Game, Board, typeof(mcts)}(params, memory, oracle, mcts, logger)
  end
end

function learning!(env::Env{G}, lp::LearningParams) where G
  # Initialize the training process
  newnn = copy(env.bestnn)
  ap = env.params.arena
  trainer, tconvert = @timed Trainer(newnn, get(env.memory), lp)
  init_loss = loss_report(trainer)
  epochs = Report.Epoch[]
  checkpoints = Report.Checkpoint[]
  # Printing utilities
  num_fmt = ".4f"
  epoch_table = Table((
    "Loss"=>num_fmt,
    "Lp"=>num_fmt,
    "Lv"=>num_fmt,
    "MaxW"=>num_fmt,
    "MeanW"=>num_fmt))
  function print_loss(lossrep, netrep, checkpoint_comment="")
    print_table_row(env.logger, epoch_table, (
      "Loss"=>lossrep.L,
      "Lp"=>(lossrep.Lp - lossrep.Hp),
      "Lv"=>lossrep.Lv,
      "MaxW"=>netrep.maxw,
      "MeanW"=>netrep.meanw),
      checkpoint_comment)
  end
  print_loss(init_loss, network_report(newnn.nn))
  # Loop state variables
  k = 1 # epoch number
  best_eval_score = ap.update_threshold
  next_nn = env.bestnn
  nn_replaced = false
  last_loss = init_loss.L
  stable_loss = false
  # Loop over epochs
  while !stable_loss && k <= lp.max_num_epochs
    ttrain = @elapsed training_epoch!(trainer)
    lossrep, tloss = @timed loss_report(trainer)
    netrep = network_report(newnn.nn)
    stable_loss = (last_loss - lossrep.L < lp.stop_loss_eps)
    push!(epochs, Report.Epoch(ttrain, tloss, lossrep, netrep))
    checkpoint_comment = ""
    if stable_loss || k % lp.epochs_per_checkpoint == 0
      # Make checkpoint
      evalrep, evaltime = @timed evaluate_oracle(G, env.bestnn, newnn, ap)
      push!(checkpoints, Report.Checkpoint(k, evaltime, evalrep))
      checkpoint_comment = "Evaluation reward: $(evalrep.average_reward)"
      # If eval is good enough, replace network
      if evalrep.average_reward >= best_eval_score
        nn_replaced = true
        next_nn = copy(newnn)
        best_eval_score = evalrep.average_reward
        checkpoint_comment *= " / Networked replaced"
        lp.stop_after_first_winner && break
      end
    end
    print_loss(lossrep, netrep, checkpoint_comment)
    k += 1
  end
  rep = Report.Learning(tconvert, init_loss, epochs, checkpoints, nn_replaced)
  return next_nn, rep
end

function self_play!(env::Env{G}, params=env.params.self_play) where G
  player = MctsPlayer(env.mcts,
    params.num_mcts_iters_per_turn,
    τ = params.temperature,
    nα = params.dirichlet_noise_nα,
    ϵ = params.dirichlet_noise_ϵ)
  games = Vector{Report.Game}(undef, params.num_games)
  for i in 1:params.num_games
    grep = self_play!(G, player, env.memory)
    games[i] = grep
  end
  return Report.SelfPlay(games)
end

function train!(
    env::Env{G},
    num_iters=env.params.num_iters
  ) where G
  num_nn_params = num_parameters(env.bestnn)
  iterations = Report.Iteration[]
  for i in 1:num_iters
    section(env.logger, 1, "Starting iteration $i")
    section(env.logger, 2, "Starting self-play")
    sprep, sptime = @timed self_play!(env, env.params.self_play)
    print(env.logger, "Number of simulated games: $(length(sprep.games))")
    avg_len = fmt(".1f", mean(g.length for g in sprep.games))
    print(env.logger, "Average game length: $avg_len")
    mem_size = format(length(env.memory), commas=true)
    print(env.logger, "Total memory size: $mem_size")
    section(env.logger, 2, "Starting learning")
    (newnn, lrep), ltime = @timed learning!(env, env.params.learning)
    push!(iterations, Report.Iteration(sptime, ltime, sprep, lrep))
    if lrep.nn_replaced
      env.bestnn = newnn
      env.mcts = MCTS.Env{G}(env.bestnn, env.params.self_play.cpuct)
    end
  end
  return Report.Training(num_nn_params, iterations)
end
