# AlphaZero.jl

This package provides a _generic_, _simple_ and _fast_ implementation of
Deepmind's AlphaZero algorithm:

* The core algorithm is only 2,000 lines of pure, hackable Julia code.
* Generic interfaces make it easy to add support for
  [new games](@ref game_interface) or new
  [learning frameworks](@ref network_interface).
* Being between one and two orders of magnitude faster than its Python alternatives,
  this implementation enables solving nontrivial games on
  a standard desktop computer with a GPU.
* The same agent can be trained on a
  [cluster of machines](https://www.youtube.com/watch?v=JVUJ5Oohuhs) as easily as on a
  single computer and without modifying a single line of code.

### Why should I care about AlphaZero?

Beyond its much publicized success in attaining superhuman level at games
such as Chess and Go, DeepMind's AlphaZero algorithm illustrates a more
general methodology of combining learning and search to explore large
combinatorial spaces effectively. We believe that this methodology can
have exciting applications in many different research areas.

### What does make this implementation fast and why does it matter?

Because AlphaZero is resource-hungry, successful open-source
implementations (such as [Leela Zero](https://github.com/leela-zero/leela-zero))
are written in low-level languages (such as C++) and optimized for highly
distributed computing environments.
This makes them hardly accessible for students, researchers and hackers.

Many simple Python implementations can be found on Github, but none of them is
able to beat a reasonable baseline on games such as _Othello_ or
_Connect Four_. As an illustration, the benchmark in the README of the
[most popular of them](https://github.com/suragnair/alpha-zero-general) only
features a _random_ baseline, along with a _greedy_ baseline that
does not appear to be significantly stronger.

AlphaZero.jl is designed to be as simple as those Python implementations.
In addition, it is between one and two orders of magnitude faster, making it possible
to solve nontrivial games on a standard desktop computer with a GPU.
This gain comes mostly from two sources:
- **Julia's inherent speed:** most machine learning algorithms do not suffer
  much from being written in Python as most of the computation happens within
  heavily optimized matrix manipulation routines. This is not the case with
  AlphaZero, where tree search is also a possible bottleneck.
- **An asynchronous simulation mechanism** that enables batching requests to the neural
  network across several simulation threads, thereby maximizing GPU utilization.

### Supporting and Citing

If you want to support this project and help it gain visibility, please consider starring
the repository. Doing well on such metrics may also help us secure academic funding in the
future. Also, if you use this software as part of your research, we would appreciate that
you include the following
[citation](https://github.com/jonathan-laurent/AlphaZero.jl/blob/master/CITATION.bib) in
your paper.

### Acknowledgements

This material is based upon work supported by the United States Air Force and
DARPA under Contract No. FA9550-16-1-0288 and FA8750-18-C-0092.
Any opinions, findings and conclusions or recommendations expressed in this material are
those of the author(s) and do not necessarily reflect the views of the United States
Air Force and DARPA.
