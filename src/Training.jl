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
  epochs = Report.Epoch[]
  checkpoints = Report.Checkpoint[]
  init_status = learning_status(trainer)
  Report.print_learning_status(env.logger, init_status)
  # Loop state variables
  k = 1 # epoch number
  best_eval_score = ap.update_threshold
  next_nn = env.bestnn
  nn_replaced = false
  last_loss = init_status.loss.L
  stable_loss = false
  # Loop over epochs
  while !stable_loss && k <= lp.max_num_epochs &&
        !(lp.stop_after_first_winner && nn_replaced)
    ttrain = @elapsed training_epoch!(trainer)
    status, tloss = @timed learning_status(trainer)
    stable_loss = (last_loss - status.loss.L < lp.stop_loss_eps)
    last_loss = status.loss.L
    push!(epochs, Report.Epoch(ttrain, tloss, status))
    comments = String[]
    # Decide whether or not to make a checkpoint
    if stable_loss || k % lp.epochs_per_checkpoint == 0
      eval, evaltime = @timed evaluate_oracle(G, env.bestnn, newnn, ap)
      push!(checkpoints, Report.Checkpoint(k, evaltime, eval))
      push!(comments, "Evaluation reward: $(eval.average_reward)")
      # If eval is good enough, replace network
      if eval.average_reward >= best_eval_score
        nn_replaced = true
        next_nn = copy(newnn)
        best_eval_score = eval.average_reward
        push!(comments, "Networked replaced")
      end
    end
    stable_loss && push!(comments, "Loss stabilized")
    Report.print_learning_status(env.logger, status, comments)
    k += 1
  end
  rep = Report.Learning(
    tconvert, init_status, epochs, checkpoints, nn_replaced)
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
    Log.section(env.logger, 1, "Starting iteration $i")
    Log.section(env.logger, 2, "Starting self-play")
    sprep, sptime = @timed self_play!(env, env.params.self_play)
    Log.section(env.logger, 2, "Starting learning")
    (newnn, lrep), ltime = @timed learning!(env, env.params.learning)
    iterrep = Report.Iteration(sptime, ltime, sprep, lrep)
    push!(iterations, iterrep)
    if lrep.nn_replaced
      env.bestnn = newnn
      env.mcts = MCTS.Env{G}(env.bestnn, env.params.self_play.cpuct)
    end
  end
  return Report.Training(num_nn_params, iterations)
end
