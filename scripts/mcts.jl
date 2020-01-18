using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "tictactoe")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

game = Game()
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}(), nworkers=1)
computer = MctsPlayer(env, niters=100, timeout=1.0, τ=0.5)

interactive!(game, computer, Human{Game}())
#explore(Explorer(computer))
