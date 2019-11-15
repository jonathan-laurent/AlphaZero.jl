using AlphaZero

include("using_game.jl")
@using_default_game

game = Game()
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
GI.interactive!(game, MCTS.AI(env, timeout=1.), GI.Human())
