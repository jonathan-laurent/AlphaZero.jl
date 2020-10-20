using AlphaZero
using ProgressMeter
using Statistics: mean

gspec = Examples.games["tictactoe"]
baseline = MinMax.Player(depth=5, amplify_rewards=true, τ=0.2)
mcts = MCTS.Env(gspec, MCTS.RolloutOracle(gspec))
mcts = MctsPlayer(mcts, niters=1000, τ=ConstSchedule(0.5))
player = TwoPlayers(mcts, baseline)

num_games = 200
bar = Progress(num_games)
rewards = map(1:num_games) do _
  trace = play_game(gspec, player)
  next!(bar)
  return total_reward(trace)
end

println("Average reward: $(mean(rewards))")
