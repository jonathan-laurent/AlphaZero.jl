module BitwiseTicTacToe

using StaticArrays

using ....BatchedEnvs
using ....Util.StaticBitArrays

export BitwiseTicTacToeEnv

const CROSS = true
const NOUGHT = false

const BitBoard = StaticBitArray{3 * 3 * 2, 1}

"""
A tictactoe environment implemented using bitwise operations
that can be run on GPU.
"""
struct BitwiseTicTacToeEnv
    board::BitBoard
    curplayer::Bool
end

BitwiseTicTacToeEnv() = BitwiseTicTacToeEnv(BitBoard(), CROSS)

BatchedEnvs.state_size(::Type{BitwiseTicTacToeEnv}) = (2 * 3 * 3,)
BatchedEnvs.num_actions(::Type{BitwiseTicTacToeEnv}) = 9

posidx(n, player) = n + 9 * player
posidx(x, y, player) = posidx(x + 3 * (y - 1), player)

function Base.show(io::IO, ::MIME"text/plain", env::BitwiseTicTacToeEnv)
    string_repr = Base.string(env)
    print(io, string_repr)
end

function Base.show(io::IO, env::BitwiseTicTacToeEnv)
    string_repr = Base.string(env)
    print(io, string_repr)
end

function Base.string(env::BitwiseTicTacToeEnv)
    X, O = CROSS, NOUGHT
    curplayer = (env.curplayer == X) ? "X" : "O"
    s = "$curplayer to play:\n\n"
    for i in 1:3
        for j in 1:3
            s *= env.board[posidx(i, j, X)] ? "X" : (env.board[posidx(i, j, O)] ? "O" : "·")
            (j < 3) && (s *= " ")
        end
        s *= "\n"
    end
    return s
end

function BatchedEnvs.act(env::BitwiseTicTacToeEnv, pos)
    board = Base.setindex(env.board, true, posidx(pos, env.curplayer))
    newenv = BitwiseTicTacToeEnv(board, !env.curplayer)
    reward = is_win(newenv, env.curplayer) ? 1 : (is_win(newenv, !env.curplayer) ? -1 : 0)
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

function BatchedEnvs.reset(::BitwiseTicTacToeEnv)
    return BitwiseTicTacToeEnv()
end

function get_player_board(env::BitwiseTicTacToeEnv, player)
    return @SVector [env.board[posidx(i, player)] for i in 1:9]
end

"""
    vectorize_state(env::BitwiseTicTacToeEnv)

Create a vectorize representation of the board.
The board is represented from the perspective of the next player to play.
It is a flatten 2x3x3 array with the following channels:
    [next player, other player]
"""
function BatchedEnvs.vectorize_state(env::BitwiseTicTacToeEnv)
    nbrd = get_player_board(env, NOUGHT)
    cbrd = get_player_board(env, CROSS)
    order = (env.curplayer == NOUGHT) ? (@SVector [nbrd, cbrd]) : (@SVector [cbrd, nbrd])
    return Float32.(reduce(vcat, order))
end


end
