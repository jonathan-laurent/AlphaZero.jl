using AlphaZero

gspec = Examples.games["tictactoe"]
mcts = MCTS.Env(gspec, MCTS.RolloutOracle(gspec))
computer = MctsPlayer(mcts, niters=1, timeout=1.0, τ=ConstSchedule(0.5))

# interactive!(gspec, computer, Human())
explore(computer, gspec)