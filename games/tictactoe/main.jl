module Tictactoe
  export Game, Board
  include("game.jl")
  module Training
    using AlphaZero
    include("params.jl")
  end
end
