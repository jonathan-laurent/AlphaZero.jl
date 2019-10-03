import AlphaZero.GI
import AlphaZero.MCTS

include("game.jl")
import .TicTacToe

Game = TicTacToe.Game
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
GI.interactive!(Game(), MCTS.AI(env, timeout=1.), GI.Human())
