"""
The default user interface for AlphaZero.

The user interface is fully separated from the core algorithm and can therefore
be replaced easily.
"""
module UserInterface

  export Log
  export Explorer, start_explorer
  export Session, resume!, save, play_interactive_game

  using AlphaZero
  import AlphaZero: Util, GameType, apply_temperature
  using AlphaZero.Util: Option

  import Plots
  import Colors
  import JSON2
  import JSON3

  using Base: @kwdef
  using Statistics: mean
  using Formatting
  using Crayons
  using Colors: @colorant_str
  using ProgressMeter

  using Serialization: serialize, deserialize

  include("log.jl")
  using .Log

  include("explorer.jl")
  include("plots.jl")
  include("json.jl")
  include("session.jl")

end
