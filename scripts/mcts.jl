using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "tictactoe")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
computer = MctsPlayer(env, niters=1, timeout=1.0, τ=ConstSchedule(0.5))

#interactive!(Game(), Human{Game}(), computer)
start_explorer(Explorer(computer))
