#####
##### Profiling utilities
#####

function profile_self_play(network, params, n)
  player = MctsPlayer(network, params)
  time = @elapsed for i in 1:n
    play(player, player)
  end
  itr = MCTS.inference_time_ratio(player.mcts)
  aed = MCTS.average_exploration_depth(player.mcts)
  return (time / n, itr, aed)
end

function profile_self_play(configs::Vector)
  log  = Logger()
  time = Log.ColType(10, x -> format("{:.1f} min", x * 100 / 60))
  itr  = Log.ColType(5,  x -> format("{}%", round(Int, x * 100)))
  expd = Log.ColType(5,  x -> format("{:.1f}", x))
  tab  = Log.Table([
    ("T100", time, s -> s[1]),
    ("ITR",  itr,  s -> s[2]),
    ("EXPD", expd, s -> s[3])])
  for (title, net, params) in configs
    profile_self_play(net, params, 1) # Compilation
    rep = profile_self_play(net, params, 10)
    Log.table_row(log, tab, rep, [title])
  end
end
