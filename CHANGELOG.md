# Changelog

## Version 0.1

Initial release, as announced on [Julia's discourse](@ref https://discourse.julialang.org/t/announcing-alphazero-jl/36877).

## Version 0.2

- Bug fix: the network policy target is not affected by the move selection temperature anymore.
  See [this post](https://discourse.julialang.org/t/announcing-alphazero-jl/36877/29?u=jonathan-laurent)
  for details.
- Introduced the `prior_temperature` MCTS parameter.
- Added a script to tune MCTS parameters.
- Tuned the MCTS parameters of the connect four agent, resulting in a significant improvement.
