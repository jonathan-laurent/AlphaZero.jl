module Tictactoe
  export GameEnv, GameSpec, Board
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
end
