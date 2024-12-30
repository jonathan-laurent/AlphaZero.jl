# Monte Carlo Tree Search (MCTS) directory structure

- [BatchedMcts.jl](BatchedMcts.jl): Contains the `BatchedMcts` module, which implements the
    batched MCTS algorithm both for CPU and GPU devices. It contains both the traditional
    MCTS implementation described in
    [[Silver et al., 2017]](https://www.nature.com/articles/nature24270), as well as the
    Gumbel-MCTS variant described in
    [[Danihelka et al., 2022]](https://openreview.net/pdf?id=bERaNdoegnO).
- [BatchedMctsAos.jl](BatchedMctsAos.jl): Contains the `BatchedMctsAos` module, which implements
    the batched MCTS algorithm as in [BatchedMcts.jl](BatchedMcts.jl), but with the slight
    difference that is uses the Array of Structures (AoS) data layout instead of the Structure
    of Arrays (SoA) data layout. This implementation hasn't been tested yet, and thus it's not
    ready for use.
- [BatchedMctsUtilities.jl](BatchedMctsUtilities.jl): Contains the `BatchedMctsUtilities` module,
    which contains definitions of important structures such as the `EnvOracle`, the MCTS
    configurations: `GumbelMctsConfig` and `AlphaZeroMctsConfig`, and some other utilities.
- [Oracles.jl](Oracles.jl): Contains the `EnvOracles` module, which defines functions to create
    a uniform and a Neural Network based `EnvOracle`.
- [SimpleMcts.jl](SimpleMcts.jl): Contains the `SimpleMcts` module, a simple, non-batched
    Gumbel-MCTS implementation mainly used for testing purposes.