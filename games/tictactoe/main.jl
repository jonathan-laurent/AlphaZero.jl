module Tictactoe
  export GameEnv, GameSpec, Board
  include("game.jl")
  module Training
    using AlphaZero
    include("params.jl")
  end
end
