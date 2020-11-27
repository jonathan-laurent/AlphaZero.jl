module GridWorld
  export GameSpec, GameEnv
  using AlphaZero
  include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
end
