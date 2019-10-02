import AlphaZero.GI
import AlphaZero.MCTS

include("game.jl")
import .TicTacToe

import Distributions: Categorical

const INITIAL_TRAINING = 100_000

const TIMEOUT = 10_000

struct MonteCarloAI <: GI.Player
  env :: MCTS.Env
  timeout :: Int
end

function GI.select_move(ai::MonteCarloAI, state)
  MCTS.explore!(ai.env, state, ai.timeout)
  actions, distr = MCTS.policy(ai.env)
  actions[rand(Categorical(distr))]
end

const Game = TicTacToe.Game
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
state = Game()
MCTS.explore!(env, state, INITIAL_TRAINING)

# MCTS.debug_tree(env)
GI.interactive!(state, MonteCarloAI(env, TIMEOUT), GI.Human())
