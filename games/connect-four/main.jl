module ConnectFour
  export Game, Board
  include("game.jl")
  module Training
    using AlphaZero
    include("params.jl")
  end
  include("solver.jl")
end
