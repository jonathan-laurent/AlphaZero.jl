module TestEnvs

using ReinforcementLearningEnvironments
using Random: MersenneTwister
using Statistics: mean

export tictactoe_draw, tictactoe_winning

function tictactoe_position(actions)
    env = TicTacToeEnv()
    for a in actions
        env(a)
    end
    return env
end

"""
Return the following simple tictactoe position (O to play).

    1 4 7    O X O
    2 5 8    . X .
    3 6 9    X . .

This position should result in a draw.
"""
tictactoe_draw() = tictactoe_position([5, 1, 3, 7, 4])

"""
Return the following simple tictactoe position (X to play).

    1 4 7    X . .
    2 5 8    . X .
    3 6 9    . O O

This position should result in a win for X.
"""
tictactoe_winning() = tictactoe_position([5, 6, 1, 9])

end
