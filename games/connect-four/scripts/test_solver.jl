using AlphaZero

include("main.jl")
using .ConnectFour

computer = ConnectFour.Solver.Player()
human = Human()

interactive!(Game(), computer, human)
