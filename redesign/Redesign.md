# AlphaZero.jl Redesign

## Redesign Objectives

- A codebase that is more accessible and easier to read
  - The file hierarchy reflects the module hierarchy
  - Literate programming using `Pollen`
  - A layered API similar to the one used in FastAI?
- Improved integration with the rest of the ecosystem:
  - Use the `ReinforcementLearning.RLBase` interface for environments
  - Use the `Logging` and `ProgressLogging` modules for logging
  - Use the `PrettyTables` and `Term` packaged for the Terminal UI
- Better support for distributed computing
  - Agent-based architecture for leveraging multiple machines and GPU (the previous version only parallelizes data generation)
- Improved performances
  - The goal is to be competitive with Fabrice Rosay's `AZ.jl` implementation (or even `AlphaGPU`).
- Improved modularity
  - Support for both AlphaZero and MuZero algorithms
  - Support for several MCTS implementations (including a batched MCTS implementation and a full-GPU implementation)
- Batteries included
  - Provide a test suite for new environments
  - Check hyperparameters consistency
  - Provide standard hyperparameter tuning utilities
  - Provide profiling utilities

## Mistakes to fix

- Having a centralized hyperparameters structure is not a good idea as it introduces a lot of coupling in the codebase and prevents switching components easily. Having JSON serialization of all hyperparameters by default is also exceedingly rigid. In general, centralizing all hyperparameters in a serializable structure should happen at a higher API layer.
- The way training statistics are logged using `Report` is too heavy. The logging library should be used for this.
- Submodules are underused, which hurts discoverability.
- Tests are lacking.
- Latency is very high, in part due to making many types unnecessarily parametric.
- Using multithreading for the workers and inference server may lead to bad performances at the GC constantly stops the world. Having multiple processes may be better (i.e. using Distributed).
- Precompilation is broken since we rely on conditional loading (for the CUDA_MEMORY_POOL and USE_KNET flags). We should use `Preferences` instead.
- To help with replicability, random number generators must be passed explicitly.

## Codebase Architecture

## Coding Style

- This codebase enforces the [Blue Style](https://github.com/invenia/BlueStyle).
- Each source file defines a submodule with the same name. The files hierarchy perfectly reflects the underlying module hierarchy.
- All leaf module names must be unique. In particular, this makes it easier to open files in editors such as VSCode without running into ambiguities.
- The tests for a leaf module with name `Module` is in `Tests/ModuleTests`. Note that the `Tests` directory has a flat structure. It also contains a `Common` submodule with definitions that are common to several test modules.
- The imports in each submodule are split in two parts: the external package imports first and then the internal submodule imports.
- We use the `Reexport` package so as to ease working with module hierarchies.
- We should make sure that the codebase can be explored using the "Jump to definition" feature of VS-Code.
- As specified by BlueStyle, multiline comments and docstrings should be wrapped at 92 lines. This is not enforced by the formatter but one can use a VSCode extension such as `Rewrap` to do this automatically (i.e. usinig the Alt-Q shortcut).
- **Unresolved:** It is still in debate whether type annotations should be used liberally or only for the purpose of dispatch. (Let's do the latter for now.)

## Testing

We make an unusual architecture decision by including all tests in the main `RLZero` package. This enables organizing the tests neatly using submodules while still benefitting from good Revise/editor support. Indeed, Revise can only track code in a package or in a single standalone script (via includet). An alternative would be to have a separate testing package but the current tooling does not make this easy.

## Setup

We are working with the master version of ReinforcementLearning.jl so you should update your Manifest accordingly:

```
] add ReinforcementLearningBase#master ReinforcementLearningEnvironments#master
```

## Workflow

We use `JuliaFormatter` to format the code on save. To do so, use the following VSCode configuration:

```json
{
    "[julia]": {
      "editor.detectIndentation": false,
      "editor.insertSpaces": true,
      "editor.tabSize": 4,
      "files.insertFinalNewline": true,
      "files.trimFinalNewlines": true,
      "files.trimTrailingWhitespace": true,
      "editor.rulers": [92],
      "editor.formatOnSave": true
    }
}
```

To run the VSCode debugger within the REPL, just write:

```julia
@run function_to_debug()
```

Also, if the "Jump to Definition" VSCode feature does not work, you may one to relaunch the "Choose Julia Env" command. This can be done by clicking on the status bar.

By executing code directly in the editor window, the whole stack trace gets highlighted in red in the editor whenever an exception is raised.

## Dev Plan

- We start implementing a minimal version of AlphaZero as it is easier:
  - Reset MCTS tree everytime for now.

## Useful Links

- [Fabrice Rosay's AlphaGPU](https://github.com/fabricerosay/AlphaGPU)
- [MuZero Pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
- [Michal Lukomski's GSOC Project](https://github.com/michelangelo21/MuZero)
- [Werner Duvaud's implementation](https://github.com/werner-duvaud/muzero-general)
- [Duvaud's Tictactoe Params](https://github.com/werner-duvaud/muzero-general/blob/master/games/tictactoe.py)

Note that in the MuZero pseudocode, they seem to be updating the network every 1000 batch updates (batches have size 2048). There are 1e6 updates in total so this makes 1000 iterations. The buffer is surprisingly small with 1e6 samples.