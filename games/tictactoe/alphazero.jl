using Revise

import AlphaZero

Revise.includet("game.jl") ; import .TicTacToe
Revise.includet("params.jl")

using Serialization: serialize, deserialize

ENV_DATA = "env.data"

CACHE = false

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
