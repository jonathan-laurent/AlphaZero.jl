using Revise
using AlphaZero
using Serialization: serialize, deserialize

DIR = "session-nets"
MEMFILE = joinpath(DIR, "mem.data")

Revise.includet("game.jl") ; import .TicTacToe ; Game = TicTacToe.Game
Revise.includet("params.jl")

params = Params(params,
  self_play=SelfPlayParams(params.self_play,
    num_games=20_000,
    reset_mcts_every=20_000),
  learning=LearningParams(params.learning,
    checkpoints=[2, 4, 8, 12, 20, 30, 40, 50, 60, 70, 80]))

function analyze_network(title, experience, network)
  env = Env{Game}(params, network, experience)
  dir = joinpath(DIR, title)
  mkpath(dir)
  session = Session(env, dir)
  AlphaZero.Log.section(session.logger, 1, "Benchmarking: $title")
  memory_report(session.env, session)
  report = learning!(session.env, session)
  plot_learning(report, session.env.params, dir)
end

if !isdir(DIR)
  # Generate the data
  mkpath(DIR)
  let session = Session(Game, Network,
      params, netparams, dir=DIR, autosave=false)
    AlphaZero.Log.section(session.logger, 1, "Generating playing experience")
    self_play!(session.env, session)
    serialize(MEMFILE, get_experience(session.env))
  end
end

mem = deserialize(MEMFILE)

analyze_network("baseline", mem,
  Network(netparams))
  
analyze_network("batch-norm", mem,
  Network(FluxNets.SimpleNetHP(netparams, width=600)))

analyze_network("full-resnet", mem, AlphaZero.ResNet{Game}(
  AlphaZero.ResNetHP(
    num_filters=64,
    num_blocks=5,
    num_policy_head_filters=32,
    num_value_head_filters=32)))
