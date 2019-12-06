# AlphaZero.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jonathan-laurent.github.io/AlphaZero.jl/dev)
[![Build Status](https://travis-ci.com/jonathan-laurent/AlphaZero.jl.svg?branch=master)](https://travis-ci.com/jonathan-laurent/AlphaZero.jl)
[![Codecov](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl)

A generic, simple and fast implementation of Deepmind's AlphaZero algorithm.

* **Generic.** Supports both Flux and Knet.
  New games can be added easily by implementing the
  [game interface](https://jonathan-laurent.github.io/AlphaZero.jl/dev/game_interface/).

* **Simple.** The core algorithm only consists in 2000 lines
  of pure Julia code.
* **Fast.** Thanks to Julia's inherent speed and to an *asynchronous*
  implementation of Monte-Carlo Tree Search, `AlphaZero.jl` is about 30x faster
  than existing python implementations
  (the most popular being [this one](https://github.com/suragnair/alpha-zero-general)).
  This makes it possible to solve nontrivial games on a standard desktop
  computer with a GPU.

## How to run

To launch a training session for the game _connect four_, just run:

```ssh
julia --project --color=yes scripts/alphazero.jl --game connect-four train
```

The system should already play at a decent level after six hours of training on
a standard desktop computer with a GPU.

To get optimal performances, it is recommended to run `AlphaZero.jl` with
Julia 1.4 (nightly), which includes a
[critical feature](https://github.com/JuliaLang/julia/pull/33448s)
that enables `CuArrays` to force incremental GC collections.
