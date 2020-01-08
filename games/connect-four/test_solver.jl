using AlphaZero

include("main.jl")
using .ConnectFour

computer = ConnectFour.Solver.Player()
human = GI.Human{Game}()

GI.interactive!(Game(), computer, human)
