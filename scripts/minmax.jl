using AlphaZero

include("game_module.jl")
@game_module SelectedGame
using .SelectedGame: Game

computer = MinMax.Player{Game}(depth=4, τ=0.2)

println("Profiling thinking time:")
@time GI.select_move(computer, Game())
println("")

GI.interactive!(Game(), computer, GI.Human{Game}())
