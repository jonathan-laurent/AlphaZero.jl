module OSpielTictactoe
  export GameSpec
  include("game.jl")
  module Training
    using ..AlphaZero
    import ..GameSpec
    include("params.jl")
 end # module Training
end # module OSpielTictactoe