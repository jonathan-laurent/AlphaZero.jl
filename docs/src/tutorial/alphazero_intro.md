# [Introduction to AlphaZero](@id alphazero_intro)

The AlphaZero algorithm elegantly combines _search_ and _learning_, which are
described in Rich Sutton's essay ["The Bitter
Lesson"](http://incompleteideas.net/IncIdeas/BitterLesson.html) as the two
fundamental pillars of AI. It augments a tree search procedure with two learnt
heuristics: one to evaluate board positions and one to concentrate branching on
moves that are not obviously wrong.

When training starts, both heuristics are initialized randomly and tree search
has only access to a meaningful signal at the level of final states, where the
game outcome is known. These heuristics are then improved iteratively
through self-play. More specifically:

- The heuristics are implemented by a **two-headed neural network**. Given
  a board position as an input, it estimates the probability for each player to
  ultimately win the game. It also provides a quantitative estimate of the
  relative quality of all available moves
  in the form of a probability distribution.
- The search component is powered by **Monte-Carlo Tree Search** (MCTS),
  which implements
  a good compromise between _breadth-first_ and _depth-first_ search and
  provides a principled way to manage the uncertainty introduced by
  the heuristics. Also, given a position, it does not return a single
  choice for a best move but rather a probability distribution over
  available moves.
- At each training iteration, AlphaZero **plays a series of games against
  itself**. The network is then updated so that it makes more accurate
  predictions about the outcome of these games. Also, the network's policy
  heuristic is updated to match the output of MCTS on all encountered positions.
  This way, MCTS can be seen as a powerful policy improvement operator.

For more details, we recommend the following resources.

### External resources

- A short and effective introduction to AlphaZero is Surag Nair's
   [excellent tutorial](https://web.stanford.edu/~surag/posts/alphazero.html).
- Our JuliaCon 2021 [talk](https://www.youtube.com/watch?v=nbLmR0aDumo)
   features a ten-minute introduction to AlphaZero and discusses some research challenges
   of using it to solve problems beyond board games.
- A good resource to learn about Monte Carlo Tree Search (MCTS) is this
   [Int8 tutorial](https://int8.io/monte-carlo-tree-search-beginners-guide/).
- Then, DeepMind's original
   [Nature paper](https://www.nature.com/articles/nature24270)
   is a nice read.
- Finally, this [series of posts](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191)
   from Oracle has been an important source of inspiration for `AlphaZero.jl`.
   It provides useful details on implementing asynchronous MCTS, along with
   an interesting discussion on hyperparameters tuning.
