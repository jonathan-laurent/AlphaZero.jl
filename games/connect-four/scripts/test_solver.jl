using AlphaZero

include("main.jl")
using .ConnectFour

computer = ConnectFour.Solver.Player()
human = Human{Game}()

interactive!(Game(), computer, human)
