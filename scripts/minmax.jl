using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

const DEPTH = parse(Int, get(ENV, "DEPTH", "5"))

computer = MinMax.Player{Game}(depth=DEPTH, amplify_rewards=true, τ=0.2)

println("Profiling thinking time:")
AlphaZero.select_move(computer, Game(), 0)
@time AlphaZero.select_move(computer, Game(), 0)
println("")

interactive!(Game(), computer, Human{Game}())
