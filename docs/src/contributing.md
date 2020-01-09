# [Contributions Guide](@id contributing)

Contributions to `AlphaZero.jl` are most welcome. Here are some contribution
ideas:

  - [Add support for a new game](@ref)
  - [Improve the user interface](@ref)
  - [Help with hyperparameter tuning](@ref)
  - Write tutorials or other learning resources based on this package
  - Design a logo
  - Add your idea here

---

#### Add support for a new game

The simplest way to contribute to `AlphaZero.jl` is probably to add support
for a new game. Games that we would be excited to see being added include:
Chess, Go 9x9, Othello, [Gobblet](https://en.wikipedia.org/wiki/Gobblet)...

Guidelines for including new games are available [here](@ref add_game).

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

#### Help with hyperparameter tuning

Because a single training session can take hours or days, it is hard for a
single person to fine-tune AlphaZero's many hyperparameters.
In an effort to tackle more and more ambitious games, it would be useful
to develop a collaborative process for running tuning experiments and share
the resulting wisdom.
