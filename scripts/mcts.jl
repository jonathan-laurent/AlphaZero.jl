using AlphaZero

gspec = Examples.experiments["tictactoe"].gspec
mcts = MCTS.Env(gspec, MCTS.RolloutOracle(gspec))
computer = MctsPlayer(mcts, niters=1, timeout=1.0, τ=ConstSchedule(0.5))

# interactive!(gspec, computer, Human())
start_explorer(Explorer(computer, gspec))