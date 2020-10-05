module Examples

using AlphaZero
using AlphaZero: NetLib

include("../games/tictactoe/main.jl")
include("../games/connect-four/main.jl")

import .Tictactoe
import .ConnectFour

const experiments_list = [
  Tictactoe.Training.experiment,
  ConnectFour.Training.experiment]

const experiments = Dict((e.name, e) for e in experiments_list)

end