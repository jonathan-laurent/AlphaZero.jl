# Project Structure

- [MCTS](MCTS): Contains all code related to the batched MCTS algorithm, such as the Environment
    Oracles, Configuration Structures, and the MCTS implementation itself. More information can be
    found on [MCTS/README.md](MCTS/README.md).
- [Networks](Networks): Contains the code related to the neural network library, such as the
    FluxNetwork interface and derived architectures such as Multi-Layer Perceptrons, Residual
    Multi-Layer Perceptrons, and more. More information can be found on
    [Networks/README.md](Networks/README.md).
- [Util](Util): Contains utility functions and types used throughout the codebase, such as
    the `StaticBitArray` structure for defining immutable envirionments, and device-agnostic
    functions for working with CPU and GPU devices.
- [BatchedEnvs.jl](BatchedEnvs.jl): Contains the `BatchedEnvs` interface, which is used to
    define environments simulated in parallel both in CPU and GPU. This interface is used by the
    MCTS algorithm to parallelize the search across a batch of environments.
- [Evaluation.jl](Evaluation.jl): Contains the `Evaluation` module, which is used to
    define the interface of evaluation functions used to evaluate AlphaZero and the Raw Neural
    Network during training, as well as provide a couple of examples of evaluation functions.
- [LoggingUtilities.jl](LoggingUtilities.jl): Contains the `LoggingUtilities` module, which
    is used to define logging functions used to log training data during training both in
    log files and to [TensorBoard](https://github.com/JuliaLogging/TensorBoardLogger.jl) logs.
- [Minimax.jl](Minimax.jl): Contains the `Minimax` module, which implements deterministic and
    stochastic versions of the [Minimax algorithm](https://en.wikipedia.org/wiki/Minimax), which
    in turn is used to create baseline agents to evaluate AlphaZero against.
- [ReplayBuffer.jl](ReplayBuffer.jl): Contains the `ReplayBuffer` module, which implements
    the ReplayBuffer structure that is used to store training data during the self-play phase,
    and sample batches of training data during the training phase.
- [RLZero.jl](RLZero.jl): Contains the `RLZero` module, which is basically the main module
    of the redesigned AlphaZero.jl package.
- [Train.jl](Train.jl): Contains the `Train` module, which implements all AlphaZero train-related
    functions, such as the self-play phase, the neural network training phase, and the evaluation
    phase.
- [TrainUtilities.jl](TrainUtilities.jl): Contains the `TrainUtilities` module, which implements
    the `TrainConfig` structure, which is the structure used to configure all the
    parameters/hyperparameters of the training process, as well as some other useful utilities.
