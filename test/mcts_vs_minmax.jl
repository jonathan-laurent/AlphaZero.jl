using AlphaZero
using ProgressMeter

# include("../games/tictactoe/main.jl")
# using .Tictactoe: Game

include("../games/connect-four/main.jl")
using .ConnectFour: Game

baseline = MinMax.Player{Game}(depth=5, amplify_rewards=true, τ=0.2)
mcts = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
player = MctsPlayer(mcts, niters=1000, τ=ConstSchedule(0.5))

num_games = 1000
bar = Progress(num_games)
avgz = AlphaZero.pit(player, baseline, num_games, gamma=1.0,
    reset_every=nothing,
    color_policy=ALTERNATE_COLORS,
    flip_probability=0.) do i, z, t
  #AlphaZero.debug_trace(t)
  next!(bar)
end
println("Average reward: $avgz")
