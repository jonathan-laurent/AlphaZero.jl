module Examples

using AlphaZero

# include("../games/tictactoe/main.jl")
# export Tictactoe

include("../games/connect-four/main.jl")
export ConnectFour

const experiments_list = [
  # Tictactoe.Training.experiment,
  ConnectFour.Training.experiment
]

const experiments = Dict((e.name, e) for e in experiments_list)

end