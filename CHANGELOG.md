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

## Version 0.3

- Generalized and simplified the game interface:
    * The symmetry assumption is removed, along with the
      board/state conceptual distinction.
    * Intermediate rewards are now supported.
    * This refactoring lays the groundwork for adding support to
      OpenSpiel.jl and CommonRLInterface.jl.
- Added a test suite to check that a given game implementation verifies all
  expected invariants.
- Simplified the MCTS implementation. It appears that a significant bug was
  fixed by doing so as the MCTS baseline now outperforms the MinMax baseline
  at Connect Four. Also, the Connect Four agent can now score a 100% win rate
  against both baselines after a couple hours of training.
