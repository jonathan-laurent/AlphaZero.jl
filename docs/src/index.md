# AlphaZero.jl

This package provides a _generic_, _simple_ and _fast_ implementation of
Deepmind's AlphaZero algorithm:

* The core algorithm is only 2,000 lines of pure, hackable Julia code.
* Generic interfaces make it easy to add support for
  [new games](@ref game_interface) or new
  [learning frameworks](@ref network_interface).
* Being about 40x faster than competing alternatives,
  this implementation enables to solve nontrivial games on a standard
  desktop computer with a GPU.

### Why should I care about AlphaZero?

Beyond its much publicized success in attaining superhuman level at games
such as Chess and Go, DeepMind's AlphaZero algorithm illustrates a more
general methodology of combining learning and search to explore large
combinatorial spaces effectively. We believe that this methodology can
have exciting applications in many different research areas.

### What does make this implementation fast and why does it matter?

Because AlphaZero is resource-hungry, successful open-source
implementations (such as
  [Leela Zero](https://github.com/leela-zero/leela-zero/tree/next/src))
are written in low-level languages (such as C++) and optimized to work
on large, distributed clusters. This makes them hardly accessible for
researchers and hackers.

Many simple Python implementations can be found on Github, but none of them is
able to beat a reasonable baseline on games such as _Connect Four_ or
_Othello_. As an illustration, the benchmark in the README of the
[most popular of them](https://github.com/suragnair/alpha-zero-general) only
features a _random_ baseline, along with a _greedy_ baseline that
does not appear to be significantly stronger.

`AlphaZero.jl` is designed to be as simple as those implementations.
In addition, it is about 40x faster, making it possible to solve nontrivial
 games on a standard desktop computer with a GPU.
This gain comes mostly from two sources:
- **Julia's inherent speed:** most machine learning algorithms do not suffer
  much from being written in python as most of the computation happens within
  heavily optimized matrix manipulation routines. This is not the case with
  AlphaZero, where tree search is also a possible bottleneck.
- **An asynchronous MCTS implementation:** even more importantly, a key
  aspect in making MCTS scale is to enable several workers to explore the
  search tree asynchronously. This is a huge win even on a single machine,
  as it enables to perform neural-network inference on large batches rather
  than evaluating board positions separately, thereby maximizing the GPU
  utilization.
