################################################################################
# Coatch.jl
################################################################################

mutable struct Env{Game, Board, Mcts}
  params :: Params
  memory :: MemoryBuffer{Board}
  bestnn :: Oracle{Game}
  mcts   :: Mcts
  function Env{Game}(params) where Game
    Board = GI.Board(Game)
    memory = MemoryBuffer{Board}(params.mem_buffer_size)
    oracle = Oracle{Game}()
    mcts = MCTS.Env{Game}(oracle, params.cpuct)
    new{Game, Board, typeof(mcts)}(params, memory, oracle, mcts)
  end
end

################################################################################

# Returns average reward for the evaluated player
# Question: when pitting network against each other,
# where does randomness come from?
# Answer: we leave a nonzero temperature
function evaluate_oracle(
    env::Env{G},
    oracle;
    τ = env.params.arena.temperature,
    num_mcts_iters = env.params.arena.num_mcts_iters_per_turn,
    num_games = env.params.arena.num_games
  ) where G
  best_mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
  new_mcts = MCTS.Env{G}(oracle, env.params.cpuct)
  best = MctsPlayer(best_mcts, num_mcts_iters, τ=τ)
  new = MctsPlayer(new_mcts, num_mcts_iters, τ=τ)
  zsum = 0.
  best_first = true
  for i in 1:num_games
    white = best_first ? best : new
    black = best_first ? new : best
    z = play_game(G, white, black)
    best_first && (z = -z)
    zsum += z
    best_first = !best_first
  end
  return zsum / num_games
end

################################################################################

function train!(
    env::Env{G},
    num_iters=env.params.num_learning_iters
  ) where G
  for i in 1:num_iters
    # Collect data using self-play
    println("Collecting data using self-play....")
    player = MctsPlayer(env.mcts,
      env.params.self_play.num_mcts_iters_per_turn,
      τ = env.params.self_play.temperature,
      nα = env.params.self_play.dirichlet_noise_nα,
      ϵ = env.params.self_play.dirichlet_noise_ϵ)
    @showprogress for i in 1:env.params.num_episodes_per_iter
      self_play!(G, player, env.memory)
    end
    # Train new network
    newnn = copy(env.bestnn)
    examples = get(env.memory)
    println("Training new network.")
    train!(newnn, examples, env.params.learning)
    z = evaluate_oracle(env, newnn)
    pwin = (z + 1) / 2
    @printf("Win rate of new network: %.0f%%\n", 100 * pwin)
    if pwin > env.params.arena.update_threshold
      env.bestnn = newnn
      env.mcts = MCTS.Env{G}(env.bestnn, env.params.cpuct)
      @printf("Replacing network.\n")
    end
  end
end

################################################################################
