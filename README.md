# AlphaZero.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jonathan-laurent.github.io/AlphaZero.jl/dev)
[![Build Status](https://travis-ci.com/jonathan-laurent/AlphaZero.jl.svg?branch=master)](https://travis-ci.com/jonathan-laurent/AlphaZero.jl)
[![Codecov](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jonathan-laurent/AlphaZero.jl)

A generic, simple and fast implementation of Deepmind's AlphaZero algorithm.

* Supports both Flux and Knet
* Relies on a fast implementation of *asynchronous* Monte-Carlo Tree Search.
* Provides a comprehensive set of debugging and profiling tools.
* New games can be added easily by implementing the [game interface](https://jonathan-laurent.github.io/AlphaZero.jl/dev/game_interface/).

## How to run

To launch a training session for game _connect four_, just run:

```ssh
julia --project --color=yes scripts/alphazero.jl --game connect-four
```
