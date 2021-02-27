# [Quick Scripts](@id scripts)

```@meta
CurrentModule = AlphaZero
```

The `AlphaZero.Scripts` module provides a quick way to execute common tasks with a single line of code. For example, starting or resuming a
training session for the connect-four example becomes as simple as executing the following command line:

```sh
julia --project -e 'using AlphaZero; Scripts.train("connect-four")'
```

The first argument of every script specifies what experiment to load. This can be specified as an object of type
[`Experiment`](@ref) or as a string from `keys(Examples.experiment)`.


### Scripts Description

```@docs
Scripts.test_game
Scripts.train
Scripts.play
Scripts.explore
```

```@docs
Scripts.dummy_run
Scripts.test_grad_updates
```