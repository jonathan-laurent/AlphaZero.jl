module Simulations

  # Players and games
  export AbstractGameEnv, AbstractGameSpec, AbstractPlayer, TwoPlayers, Trace
  export think, select_move, reset_player!, player_temperature, apply_temperature
  export play_game, interactive!, total_reward
  export MctsPlayer, RandomPlayer, EpsilonGreedyPlayer, NetworkPlayer, Human
  export ColorPolicy, ALTERNATE_COLORS, BASELINE_WHITE, CONTENDER_WHITE

  using AlphaZero.GameInterface
  using AlphaZero.Network
  using AlphaZero.Batchifier
  using AlphaZero.Util: Option, apply_temperature
  using AlphaZero: Util, GI, Network, Batchifier, MCTS

  include("trace.jl")
  include("play.jl")

end