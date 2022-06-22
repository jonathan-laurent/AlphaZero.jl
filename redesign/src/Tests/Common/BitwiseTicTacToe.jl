module BitwiseTicTacToe

using ....BatchedEnvs
using ....Util.StaticBitArrays

export BitwiseTicTacToeEnv

const CROSS = true
const NOUGHT = false

const BitBoard = StaticBitArray{18,1}

"""
A tictactoe environment implemented using bitwise operations
that can be run on GPU.
"""
struct BitwiseTicTacToeEnv
    board::BitBoard
    curplayer::Bool
end

BitwiseTicTacToeEnv() = BitwiseTicTacToeEnv(BitBoard(), CROSS)

BatchedEnvs.num_actions(::BitwiseTicTacToeEnv) = 9

posidx(n, player) = n + 9 * player
posidx(x, y, player) = posidx(x + 3 * (y - 1), player)

function Base.show(io::IO, ::MIME"text/plain", env::BitwiseTicTacToeEnv)
    for i in 1:3
        for j in 1:3
            if env.board[posidx(i, j, CROSS)]
                print(io, "X ")
            elseif env.board[posidx(i, j, NOUGHT)]
                print(io, "O ")
            else
                print(io, ". ")
            end
        end
        print(io, "\n")
    end
end

function BatchedEnvs.act(env::BitwiseTicTacToeEnv, pos)
    board = Base.setindex(env.board, true, posidx(pos, env.curplayer))
    newenv = BitwiseTicTacToeEnv(board, !env.curplayer)
    if is_win(newenv, env.curplayer)
        reward = 1.0
    elseif is_win(newenv, !env.curplayer)
        reward = -1.0
    else
        reward = 0.0
    end
    return newenv, (; reward, switched=true)
end

function BatchedEnvs.valid_action(env::BitwiseTicTacToeEnv, pos)
    return !env.board[posidx(pos, CROSS)] && !env.board[posidx(pos, NOUGHT)]
end

function full_board(env::BitwiseTicTacToeEnv)
    return !any(BatchedEnvs.valid_action(env, pos) for pos in 1:9)
end

function is_win(env::BitwiseTicTacToeEnv, player)
    at(i, j) = env.board[posidx(i, j, player)]
    return begin
        at(1, 1) & at(1, 2) & at(1, 3) ||
            at(2, 1) & at(2, 2) & at(2, 3) ||
            at(3, 1) & at(3, 2) & at(3, 3) ||
            at(1, 1) & at(2, 1) & at(3, 1) ||
            at(1, 2) & at(2, 2) & at(3, 2) ||
            at(1, 3) & at(2, 3) & at(3, 3) ||
            at(1, 1) & at(2, 2) & at(3, 3) ||
            at(1, 3) & at(2, 2) & at(3, 1)
    end
end

function BatchedEnvs.terminated(env::BitwiseTicTacToeEnv)
    return full_board(env) || is_win(env, CROSS) || is_win(env, NOUGHT)
end

end
