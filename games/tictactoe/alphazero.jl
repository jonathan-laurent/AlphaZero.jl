import AlphaZero

using CuArrays

CuArrays.allowscalar(false)

include("game.jl") ; import .TicTacToe

using Serialization: serialize, deserialize

const ENV_DATA = "env.data"

const CACHE = false

netparams = AlphaZero.SimpleNetParams(
  width=300)

arena = AlphaZero.ArenaParams(
  reset_mcts=true,
  update_threshold=(2 * 0.55 - 1),
  num_mcts_iters_per_turn=20,
  num_games=200)

self_play = AlphaZero.SelfPlayParams(
  num_games=500,
  num_mcts_iters_per_turn=100)

learning = AlphaZero.LearningParams()

params = AlphaZero.Params(
  arena=arena,
  self_play=self_play,
  learning=learning,
  num_iters=2,
  num_game_stages=9)

if !CACHE || !isfile(ENV_DATA)
  network = AlphaZero.SimpleNet{TicTacToe.Game, netparams}()
  env = AlphaZero.Env{TicTacToe.Game}(params, network)
  AlphaZero.train!(env)
  println("\n")
  serialize(ENV_DATA, env)
else
  env = deserialize(ENV_DATA)
end

println("Launching explorer.")
explorer = AlphaZero.Explorer(env, TicTacToe.Game())
AlphaZero.launch(explorer)
