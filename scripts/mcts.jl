using AlphaZero

include("games.jl")
const GAME = get(ENV, "GAME", "tictactoe")
const SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

game = Game()
env = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
computer = MctsPlayer(env, niters=1000, timeout=1., τ=0.)
interactive!(game, computer, Human{Game}())
