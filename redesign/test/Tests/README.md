# Tests Structure

- [Common](Common): Contains environment implementations, heuristic and evaluation
    functions. More details can be found in the [README](Common/README.md) file.
- [EnvTests](EnvTests): Contains tests for the environments defined in
    [Common/Envs](Common/Envs).
- [MctsTests](MctsTests): Contains all the MCTS-related tests.
- [NetworksTests](NetworksTests): Contains all the neural network-related tests.
- [BatchedEnvTests.jl](BatchedEnvTests.jl): Contains generic tests for the batched
    environments following the [BatchedEnvs/jl](../../src/BatchedEnvs.jl) interface.
- [MinimaxTests.jl](MinimaxTests.jl): Contains tests for the minimax algorithm.
- [ReplayBufferTests.jl](ReplayBufferTests.jl): Contains tests for the episode and
    replay buffers.
- [Tests.jl](Tests.jl): Main script where all the tests are launched.
- [TrainTests.jl](TrainTests.jl): Contains tests for the training loop.
- [UtilsTests.jl](UtilsTests.jl): Contains tests for the utility functions.