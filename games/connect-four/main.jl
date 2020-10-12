module ConnectFour
  export GameSpec, GameEnv, Board
  include("game.jl")
  module Training
    using AlphaZero
    using AlphaZero.Core
    using AlphaZero.Network
    using AlphaZero.Benchmark
    using AlphaZero.Experiments
    using AlphaZero: NetLib
    import ..GameSpec
    include("params.jl")
  end
  include("solver.jl")
end
