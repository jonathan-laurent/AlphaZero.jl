module ConnectFour
  export GameSpec, GameEnv, Board
  include("game.jl")
  module Training
    using AlphaZero
    using AlphaZero: NetLib
    import ..GameSpec
    include("params.jl")
  end
  include("solver.jl")
end
