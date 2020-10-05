module Tictactoe
  export GameEnv, GameSpec, Board
  include("game.jl")
  module Training
    using AlphaZero
    using AlphaZero: NetLib
    import ..GameSpec
    include("params.jl")
  end
end
