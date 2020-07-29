using AlphaZero
using ProgressMeter
using Statistics: mean

# include("../games/tictactoe/main.jl")
# using .Tictactoe: Game

include("../games/connect-four/main.jl")
using .ConnectFour: Game

baseline = MinMax.Player{Game}(depth=5, amplify_rewards=true, τ=0.2)
mcts = MCTS.Env{Game}(MCTS.RolloutOracle{Game}())
mcts = MctsPlayer(mcts, niters=1000, τ=ConstSchedule(0.5))
player = TwoPlayers(mcts, baseline)

num_games = 200
bar = Progress(num_games)
rewards = map(1:num_games) do _
  trace = play_game(player)
  next!(bar)
  return total_reward(trace)
end

println("Average reward: $(mean(rewards))")
