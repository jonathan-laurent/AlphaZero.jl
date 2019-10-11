netparams = AlphaZero.SimpleNetParams(
  width=300,
  depth_common=3)

arena = AlphaZero.ArenaParams(
  reset_mcts=true,
  update_threshold=(2 * 0.55 - 1),
  num_mcts_iters_per_turn=20,
  num_games=200)

self_play = AlphaZero.SelfPlayParams(
  num_games=500,
  num_mcts_iters_per_turn=100)

learning = AlphaZero.LearningParams(
  use_gpu=true)

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=3,
  num_game_stages=9)
