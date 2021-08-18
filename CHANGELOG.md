# Changelog

## Version 0.1

Initial release, as announced on [Julia's discourse](https://discourse.julialang.org/t/announcing-alphazero-jl/36877).

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

## Version 0.4

- Added support for CommonRLInterface.jl.
- Added a grid-world MDP example illustrating this new interface.
- Added support for distributed training: it is now equally easy to train an agent on
  a cluster of machines than on a single computer.
- Replaced the async MCTS implementation by a more straightforward synchronous
  implementation. Network inference requests are now batched across game simulations.
- Added the Experiment and Scripts module to simplify common tasks.

## Version 0.5

- Improved the inference server so that it is now possible to keep MCTS workers
  running while a batch of requests is being processed by the GPU. Concretely,
  this translates into `SimParams` now having two separate `num_workers` and
  `batch_size` parameters.
- The inference server is now spawned on a separate thread to ensure minimal latency.

Together, the two aforementioned improvements result in a 30% global speedup on the
connect-four benchmark.

## Version 0.5.2

- Add a wrapper to use AlphaZero.jl with OpenSpiel.jl.
- Add an OpenSpiel tic-tac-toe example.