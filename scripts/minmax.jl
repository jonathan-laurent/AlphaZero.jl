using AlphaZero

include("games.jl")
GAME = get(ENV, "GAME", "connect-four")
SelectedGame = GAME_MODULE[GAME]
using .SelectedGame: GameSpec

const DEPTH = parse(Int, get(ENV, "DEPTH", "5"))

computer = MinMax.Player(depth=DEPTH, amplify_rewards=true, τ=0.2)

game = GI.init(GameSpec())
println("Profiling thinking time:")
AlphaZero.select_move(computer, game, 0)
@time AlphaZero.select_move(computer, game, 0)
println("")

interactive!(game, computer, Human())
