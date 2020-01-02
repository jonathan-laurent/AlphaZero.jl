using AlphaZero

include("games.jl")
const GAME = get(ENV, "GAME", "tictactoe")
const SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

game = Game()
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
GI.interactive!(game, MCTS.AI(env, timeout=1.), GI.Human{Game}())
