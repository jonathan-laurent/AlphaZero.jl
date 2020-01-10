# [Contributions Guide](@id contributions_guide)

Contributions to `AlphaZero.jl` are most welcome. Here are some contribution
ideas:

  - [Add support for a new game](@ref)
  - [Help with hyperparameter tuning](@ref)
  - [Improve the user interface](@ref)
  - [Develop support for a more general game interface](@ref)
  - Write tutorials or other learning resources based on this package
  - Design a logo

Also, there are many small improvements and variations that
can be built on top of this implementation and that would make for nice
ML projects. Here are a few examples:

  - Add a resignation mechanism to speed-up self-play
  - Give more weight to recent samples during learning
  - Use rollouts in addition to the network's value head to evaluate positions
    (as is done by AlphaGo Lee)
  - Implement the alternate training target proposed [here](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)

Please do not hesitate to open a Github
[issue](https://github.com/jonathan-laurent/AlphaZero.jl/issues) to share any
any idea, feedback or suggestion.

---

#### Add support for a new game

The simplest way to contribute to `AlphaZero.jl` is to add support
for new games. Interesting candidates include:
Othello, [Gobblet](https://en.wikipedia.org/wiki/Gobblet), Go 9x9, Chess...

Guidelines for including new games are available [here](@ref add_game).

---

#### Help with hyperparameter tuning

Because a single training session can take hours or days, it is hard for a
single person to fine-tune AlphaZero's many hyperparameters.
In an effort to tackle more and more ambitious games, it would be useful
to develop a collaborative process for running tuning experiments and share
the resulting wisdom.

---

#### Improve the user interface

An effort has been made in designing `AlphaZero.jl` to separate the
user interface code from the core logic (see [`AlphaZero.Handlers`](@ref)).
We would be interested in seeing alternative user interfaces being developed.
In particular, using something like TensorBoard for logging and/or profiling
might be nice.

To add a new interface option, we recommend that you proceed as follows:
  1. Create a new folder in `src/ui` to put all your code.
  2. Create a new module that exports an equivalent of the
     [`AlphaZero.Session`](@ref) type.

---

#### Develop support for a more general game interface

A first step may be to add support for nonsymmetric games. Then, a more
ambitious goal would be to add support for games with imperfect information.
Note that how to best adapt the AlphaZero approach to those games is pretty much
an open question.

On the engineering side, it may be nice to replace the current
[game interface](@ref game_interface) by something more standard such as
[OpenSpiel](https://github.com/deepmind/open_spiel), for which a Julia
[wrapper](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) is
currently being developed. Doing so would give `AlphaZero.jl` access to many
interesting game environments for free.
