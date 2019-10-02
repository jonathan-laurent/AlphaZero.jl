import AlphaZero

include("game.jl") ; import .TicTacToe

using Serialization: serialize, deserialize

const ENV_DATA = "env.data"

const CACHE = false

arena = AlphaZero.ArenaParams(
  update_threshold=0.55,
  num_mcts_iters_per_turn=20,
  num_games=100)

self_play = AlphaZero.SelfPlayParams(
  num_mcts_iters_per_turn=100)

learning = AlphaZero.LearningParams(
  num_batches=1000)

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  num_learning_iters=3,
  num_episodes_per_iter=100)

if !CACHE || !isfile(ENV_DATA)
  env = AlphaZero.Env{TicTacToe.Game}(params)
  AlphaZero.train!(env)
  serialize(ENV_DATA, env)
else
  env = deserialize(ENV_DATA)
end

explorer = AlphaZero.Explorer(env, TicTacToe.Game())
AlphaZero.launch(explorer)
