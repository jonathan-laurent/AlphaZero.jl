ENV["JULIA_CUDA_MEMORY_POOL"] = "split"

using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: GameSpec

gspec = GameSpec()
game = GI.init(gspec)
env = MCTS.Env(gspec, MCTS.RolloutOracle(gspec))
computer = MctsPlayer(env, niters=1, timeout=1.0, τ=ConstSchedule(0.5))

# interactive!(game, computer, Human())
start_explorer(Explorer(computer, game))
