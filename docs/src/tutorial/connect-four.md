# [Learning to Play Connect Four](@id connect-four)

In this section, we discuss how to use `AlphaZero.jl` to learn to play
_Connect Four_ without any form of supervision or prior knowledge.
Although the game has been [solved](https://connect4.gamesolver.org/) exactly
with Alpha-beta pruning using domain-specific heuristics and optimizations, it
is still a great challenge for reinforcement learning.[^1]

[^1]:
    To the best of our knowledge, none of the many existing Python
    implementations of AlphaZero are able to learn a player that beats a simple
    minmax baseline (that plans at depth at least 2) on a single desktop computer.

## Setup

To replicate the experiment in this tutorial, we recommend having a CUDA
compatible GPU with 6GB of memory or more. Each training iteration took about
one hour and a half on a standard desktop computer with an Intel Core i5 9600K
processor and an Nvidia RTX 2070 GPU.

!!! note
    To get optimal performances, it is also recommended to use
    `AlphaZero.jl` with Julia 1.4 (nightly), which includes a
    [critical feature](https://github.com/JuliaLang/julia/pull/33448)
    that enables `CuArrays` to force incremental GC collections.

To download `AlphaZero.jl` and start a new training session,
just run the following:

```sh
git clone https://github.com/jonathan-laurent/AlphaZero.jl.git
cd AlphaZero.jl
julia --project -e "import Pkg; Pkg.instantiate()"
julia --project --color=yes scripts/alphazero.jl --game connect-four train
```

Instead of using the the `alphazero.jl` script, one can also run the following
into the Julia REPL:

```julia
ENV["CUARRAYS_MEMORY_POOL"] = "split"

using AlphaZero

include("games/connect-four/main.jl")
using .ConnectFour: Game, Training

const SESSION_DIR = "sessions/connect-four"

session = AlphaZero.Session(
    Game,
    Training.Network{ConnectFour.Game},
    Training.params,
    Training.netparams,
    benchmark=Training.benchmark,
    dir=SESSION_DIR)

resume!(session)
```

The first line configures CuArrays to use a splitting memory pool, which
performs better than the default binned pool on AlphaZero's workload as it
does not require to run the garbage collector as frequently. Then, a new
AlphaZero [session](@ref sessions) is created with the following arguments:

| Argument             | Description                                                                     |
|:---------------------|:--------------------------------------------------------------------------------|
| `Game`               | Game type, which implements the [game interface](@ref game_interface).          |
| `Training.Network`   | Network type, which implements the [network interface](@ref network_interface). |
| `Training.params`    | Training [parameters](@ref params).                                             |
| `Training.netparams` | Network hyperparameters.                                                        |
| `Training.benchmark` | [Benchmark](@ref benchmark) that is run between training iterations.            |
| `SESSION_DIR`        | Directory in which all session files are saved.                                 |

In the following sections, we discuss some of those arguments in more details.

### Neural Network

Here, we use the same architecture
that is deployed in AlphaGo Zero, namely a two-headed convolutional resnet
with batch normalization. However, our network is
is smaller in size as it only features 7 blocks (instead of 20)
and 128 convolutional filters per layer (instead of 256), resulting in about
2.5M parameters (instead of 90M).

### Training Parameters

Here, we simulate 5000 games of
  self-play per iteration, using 400 MCTS iterations per move. For comparison,
  the original AlphaGo Zero plays 25,000 games of self-play per iteration,
  using 1600 MCTS iterations per move.

| Implementation | Games per iteration | Moves per Game | Sims per move | Inference cost |
|----------------|---------------------|----------------|---------------|----------------|
| AlphaGo Zero   |              25,000 |            200 |          1600 |            x40 |
| `AlphaZero.jl` |               5,000 |             30 |           400 |             x1 |
| Python impl    |                 100 |             30 |            25 |             x1 |

### Benchmarks

## Results

![Session CLI](../assets/img/session-ui-short.png)
