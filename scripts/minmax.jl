using AlphaZero

include("games.jl")
const GAME = get(ENV, "GAME", "connect-four")
const SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: Game

computer = MinMax.Player{Game}(depth=4, τ=0.2)

println("Profiling thinking time:")
@time AlphaZero.select_move(computer, Game())
println("")

interactive!(Game(), computer, Human{Game}())
