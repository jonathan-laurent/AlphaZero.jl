using AlphaZero

depth = 5
gspec = Examples.games["tictactoe"]
computer = MinMax.Player(depth=depth, amplify_rewards=true, τ=0.2)
interactive!(gspec, computer, Human())