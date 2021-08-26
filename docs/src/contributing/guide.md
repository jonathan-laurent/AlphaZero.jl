# [Contribution Guide](@id contributions_guide)

Contributions to AlphaZero.jl are most welcome. Here are some contribution
ideas:

  - [Solve new games](@ref)
  - [Help with hyperparameter tuning](@ref)
  - [Improve the user interface](@ref improve_ui)
  - Write tutorials or other learning resources based on this package

We also believe that AlphaZero.jl can be made even faster
without adding too much complexity to the codebase.
Here are suggestions to make this happen:
  - Accelerate network inference by adding support for FP16 or Int8 quantization
  - Accelerate network inference with [Torch.jl](https://github.com/FluxML/Torch.jl)
  - Enable data generation, network updates and checkpoint evaluations
    to be run in parallel

Finally, there are many small improvements and variations that
can be built on top of this implementation and that would make for nice
ML projects. Here are a few examples:

  - Implement and compare different [First-Play Urgency](https://lczero.org/play/flags/) strategies
  - Implement tools to automate and help with hyperparameter tuning
  - Add a resignation mechanism to speed-up self-play
  - Give more weight to recent samples during learning
  - Use rollouts in addition to the network's value head to evaluate positions
    (as is done by AlphaGo Lee)
  - Use supervised learning to initialize the network based on a set of games
    played by humans
  - Implement the alternate training target proposed [here](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
  - Implement some of the improvements introduced in the [KataGo](https://arxiv.org/pdf/1902.10565.pdf) paper


You may also want to have a look at our JSOC (Julia Summer of
Code) [project page](https://julialang.org/jsoc/gsoc/flux/#alphazerojl).

Please do not hesitate to open a Github
[issue](https://github.com/jonathan-laurent/AlphaZero.jl/issues) to share
any idea, feedback or suggestion.

---

#### Solve new games

The simplest way to contribute to AlphaZero.jl is to demonstrate it on
new games. Interesting candidates include:
Othello, [Gobblet](https://en.wikipedia.org/wiki/Gobblet), Go 9x9, Chess...
A nice first-time contribution may also be to provide an example of using AlphaZero.jl in conjunction with [OpenSpiel.jl](https://openspiel.readthedocs.io/en/latest/julia.html).

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

An effort has been made in designing AlphaZero.jl to separate the user interface code from
the core logic (see [`AlphaZero.Handlers`](@ref)). We would be interested in seeing
alternative user interfaces being developed. In particular, using something like
[TensorBoardLogger](https://github.com/PhilipVinc/TensorBoardLogger.jl) or
[Dash](https://github.com/plotly/Dash.jl) for logging and/or profiling might be nice.
