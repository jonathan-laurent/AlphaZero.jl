# [Contribution Guide](@id contributions_guide)

Contributions to `AlphaZero.jl` are most welcome. Here are some contribution
ideas:

  - [Add support for a new game](@ref)
  - [Help with hyperparameter tuning](@ref)
  - [Improve the user interface](@ref improve_ui)
  - [Develop support for a more general game interface](@ref contrib_generalize)
  - Write tutorials or other learning resources based on this package
  - Design a logo

Also, there are many small improvements and variations that
can be built on top of this implementation and that would make for nice
ML projects. Here are a few examples:

  - Add a resignation mechanism to speed-up self-play
  - Give more weight to recent samples during learning
  - Use rollouts in addition to the network's value head to evaluate positions
    (as is done by AlphaGo Lee)
  - Use supervised learning to initialize the network based on a set of games
    played by humans
  - Implement the alternate training target proposed [here](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)

Please do not hesitate to open a Github
[issue](https://github.com/jonathan-laurent/AlphaZero.jl/issues) to share
any idea, feedback or suggestion.

---

#### Add support for a new game

The simplest way to contribute to `AlphaZero.jl` is to demonstrate it on
new games. Interesting candidates include:
Othello, [Gobblet](https://en.wikipedia.org/wiki/Gobblet), Go 9x9, Chess...

Guidelines for adding support for new games are available [here](@ref add_game).

---

#### Help with hyperparameter tuning

A good place to start would be to experiment with the parameters of
the Connect Four agent discussed in the [tutorial](@ref connect_four),
as it went through little tuning and can probably be improved
significantly. Any kind of hyperparameters study would be extremely valuable
in getting a better understanding of AlphaZero's training process.

More generally, as a training session can take hours or days,
it is hard for a single person to fine-tune AlphaZero's many hyperparameters.
In an effort to tackle more and more ambitious games, it would be useful to
come up with a collaborative process for running tuning experiments and share
the resulting insights.

---

#### [Improve the user interface](@id improve_ui)

An effort has been made in designing `AlphaZero.jl` to separate the
user interface code from the core logic (see [`AlphaZero.Handlers`](@ref)).
We would be interested in seeing alternative user interfaces being developed.
In particular, using something like TensorBoard for logging and/or profiling
might be nice.

---

#### [Develop support for a more general game interface](@id contrib_generalize)

A first step may be to add support for nonsymmetric games and games with more than two players.
A more ambitious goal would be to add support for games with imperfect
information. Note that how to best adapt the AlphaZero approach to those games
is pretty much an open question though.

On the engineering side, it may be nice to replace the current
[game interface](@ref game_interface) by something more standard such as
[OpenSpiel](https://github.com/deepmind/open_spiel), for which a Julia
[wrapper](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) is
currently being developed. Doing so would give `AlphaZero.jl` access to many
interesting game environments for free.
