using AlphaZero

include("game_module.jl")
@game_module SelectedGame
using .SelectedGame: Game

game = Game()
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
GI.interactive!(game, MCTS.AI(env, timeout=1.), GI.Human{Game}())
