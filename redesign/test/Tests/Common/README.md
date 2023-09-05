# Common Structure

- [Envs](Envs): Contains the GPU-friendly implementations for the following environments:
    - [RandomWalk1D](Envs/BitwiseRandomWalk1D.jl)
    - [TicTacToe](Envs/BitwiseTicTacToe.jl)
    - [ConnectFour](Envs/BitwiseConnectFour.jl)
- [Evaluation](Evaluation): Contains two sub-directories:
    - [Heuristics](Evaluation/Heuristics): Contains heuristic functions for some
        environments defined in [Envs](Envs).
    - [EvaluationFunctions](Evaluation/EvaluationFunctions): Contains evaluation functions
        that can be run during training to assess agents, for all the environments defined
        in [Envs](Envs).