# [Codebase Overview](@id codebase_overview)

The `AlphaZero.jl` codebase is designed to be easy to understand and
experiment with and so we hope that going through it can be enlightening.

We recommend looking at the following in this order:

## Investigate the MCTS implementation

- [GameInterface](@ref game_interface):
- [MCTS](@ref mcts): before trying to understand AlphaZero,
- [Training Parameters](@ref params): in file `src/params.jl`, we explain
   and define the different parameters underlying the AlphaZero training
   process. When possible, we provided the values used in DeepMind's original
   paper. The values used in our Connect Four [tutorial](@ref connect_four) can
   be found in `games.connect-four/params.jl`.
