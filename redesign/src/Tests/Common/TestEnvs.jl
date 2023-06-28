module TestEnvs

using Random: MersenneTwister
using ReinforcementLearningBase: RLBase
using ReinforcementLearningEnvironments: TicTacToeEnv, RandomWalk1D
using Statistics: mean

using ....BatchedEnvs
using ..BitwiseTicTacToe
using ..BitwiseRandomWalk1D

export random_walk_losing, bitwise_random_walk_losing, random_walk_winning,
    bitwise_random_walk_winning
export tictactoe_draw, bitwise_tictactoe_draw, tictactoe_winning, bitwise_tictactoe_winning


function random_walk_1d_position(actions)
    env = RandomWalk1D()
    for a in actions
        RLBase.act!(env, a)
    end
    return env
end

function bitwise_random_walk_1d_position(actions)
    env = BitwiseRandomWalk1DEnv()
    for a in actions
        env, _ = act!(env, a)
    end
    return env
end

function tictactoe_position(actions)
    env = TicTacToeEnv()
    for a in actions
        RLBase.act!(env, a)
        RLBase.next_player!(env)
    end
    return env
end

function bitwise_tictactoe_position(actions)
    env = BitwiseTicTacToeEnv()
    for a in actions
        env, _ = RLBase.act!(env, a)
    end
    return env
end


"""
Return the following simple random walk position.

    · X · · · · · · · ·

This position is "closer" to losing than winning.
"""
random_walk_losing() = random_walk_1d_position([1, 1, 1])
bitwise_random_walk_losing() = bitwise_random_walk_1d_position([1, 1, 1])

"""
Return the following simple random walk position.

    · · · · · · · X · ·

This position is "closer" to winning than losing.
"""
random_walk_winning() = random_walk_1d_position([2, 2, 2])
bitwise_random_walk_winning() = bitwise_random_walk_1d_position([2, 2, 2])


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
