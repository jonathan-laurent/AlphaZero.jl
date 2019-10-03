import AlphaZero.GI
#import AlphaZero.MCTS

include("game.jl")
import .Mancala

# Late game (for debug purposes)
# game = Game(Board([10, 20], [[1 0 0 0 0 0]; [1 0 2 0 2 0]]), true)

game = Mancala.Game()
GI.interactive!(game, GI.Human(), GI.Human())
