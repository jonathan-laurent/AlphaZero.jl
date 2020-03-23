# [Learning to Play Connect Four](@id connect_four)

In this section, we discuss how to use `AlphaZero.jl` to learn to play
_Connect Four_ without any form of supervision or prior knowledge.
Although the game has been [solved](https://connect4.gamesolver.org/) exactly
with Alpha-beta pruning using domain-specific heuristics and optimizations, it
is still a great challenge for reinforcement learning.[^1]

[^1]:
    To the best of our knowledge, none of the many existing Python
    implementations of AlphaZero are able to learn a player that beats a
    minmax baseline that plans at depth 2 (on a single desktop computer).

In this tutorial, we are going to:

  1. Show you how to train a Connect Four agent on your own machine using
     `AlphaZero.jl`.
  2. Analyze the  

## Running a Training Session

## Hyperparameters

Depending on your available computing power, you may want to adjust
some of these hyperparameters.
