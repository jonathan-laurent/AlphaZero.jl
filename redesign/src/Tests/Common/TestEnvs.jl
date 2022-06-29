module TestEnvs

using ReinforcementLearningEnvironments
using Random: MersenneTwister
using Statistics: mean

using ....BatchedEnvs
using ..BitwiseTicTacToe

export tictactoe_draw, tictactoe_winning, bitwise_tictactoe_draw, bitwise_tictactoe_winning

function tictactoe_position(actions)
    env = TicTacToeEnv()
    for a in actions
        env(a)
    end
    return env
end

function bitwise_tictactoe_position(actions)
    env = BitwiseTicTacToeEnv()
    return act(env, actions)[1]
end

"""
Return the following simple tictactoe position (O to play).

    1 4 7    O X O
    2 5 8    . X .
    3 6 9    X . .

This position should result in a draw.
"""
tictactoe_draw() = tictactoe_position([5, 1, 3, 7, 4])
bitwise_tictactoe_draw() = bitwise_tictactoe_position([5, 1, 3, 7, 4])

"""
Return the following simple tictactoe position (X to play).

    1 4 7    X . .
    2 5 8    . X .
    3 6 9    . O O

This position should result in a win for X.
"""
tictactoe_winning() = tictactoe_position([5, 6, 1, 9])
bitwise_tictactoe_winning() = bitwise_tictactoe_position([5, 6, 1, 9])

end
