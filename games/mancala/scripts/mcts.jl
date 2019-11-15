import AlphaZero.GI
import AlphaZero.MCTS

include("../game.jl")
using .Mancala

# Late game (for debug purposes)
game = Game(Board([11, 11], [[1 2 4 0 0 0]; [1 2 4 0 0 0]]), true)
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
GI.interactive!(game, MCTS.AI(env, timeout=1.), GI.Human())
