Network = AlphaZero.SimpleNet{Game}

netparams = AlphaZero.SimpleNetHyperParams(
  width=300,
  depth_common=3)

# Evaluate with 0 MCTS iterations
arena = AlphaZero.ArenaParams(
  num_games=1000,
  update_threshold=(2 * 0.55 - 1),
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=0))

self_play = AlphaZero.SelfPlayParams(
  num_games=50,
  mcts = AlphaZero.MctsParams(
    num_iters_per_turn=10))

learning = AlphaZero.LearningParams()

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=3,
  num_game_stages=9)
