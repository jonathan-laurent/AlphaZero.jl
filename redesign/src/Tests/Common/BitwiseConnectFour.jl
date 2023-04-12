module BitwiseConnectFour

    using ....BatchedEnvs
    using ....Util.StaticBitArrays

    export BitwiseConnectFourEnv

    const CROSS = true
    const NOUGHT = false

    """
    Bitboard representation:

     1 -  7  ---->   · · · · · · ·       · · · · · · ·   <---  43 - 49
     8 - 14  ---->   · · · · · · ·       · · · · · · ·   <---  50 - 56
    15 - 21  ---->   · · · · · · ·       · · · · · · ·   <---  57 - 63
    22 - 28  ---->   · · · · · · ·       · · · · · · ·   <---  64 - 70
    29 - 35  ---->   · · · · · · ·       · · · · · · ·   <---  71 - 77
    36 - 42  ---->   · · · · · · ·       · · · · · · ·   <---  78 - 84

                     NOUGHT PLAYER       CROSS PLAYER
    """
    const bitboard = StaticBitArray{42 * 2, 2}

    """
    A connect-four environment implemented using bitwise operations
    that can be run on GPU.
    """
    struct BitwiseConnectFourEnv
        board::bitboard
        curplayer::Bool
    end

    function BitwiseConnectFourEnv()
        return BitwiseConnectFourEnv(bitboard(), CROSS)
    end

    BatchedEnvs.num_actions(::BitwiseConnectFourEnv) = 7

    posidx(n, player) = n + 42 * player
    posidx(x, y, player) = posidx(7 * (x - 1) + y, player)

    function Base.show(io::IO, ::MIME"text/plain", env::BitwiseConnectFourEnv)
        for i in 1:6
            for j in 1:7
                if env.board[posidx(i, j, CROSS)]
                    print(io, "X ")
                elseif env.board[posidx(i, j, NOUGHT)]
                    print(io, "O ")
                else
                    print(io, "· ")
                end
            end
            println(io)
        end
    end

    function Base.show(io::IO, env::BitwiseConnectFourEnv)
        Base.show(io, MIME("text/plain"), env)
    end

    function BatchedEnvs.act(env::BitwiseConnectFourEnv, action)
        at(i, j, player) = env.board[posidx(i, j, player)]

        curr_row = 6
        while at(curr_row, action, env.curplayer) ||
              at(curr_row, action, !env.curplayer)
            curr_row -= 1
        end

        board = Base.setindex(env.board, true, posidx(curr_row, action, env.curplayer))
        newenv = BitwiseConnectFourEnv(board, !env.curplayer)
        if is_win(newenv, !env.curplayer)
            reward = -1.0
        else
            reward = 0.0
        end

        return newenv, (; reward, switched=true)
    end

    function BatchedEnvs.valid_action(env::BitwiseConnectFourEnv, action)
        return begin
            !env.board[posidx(1, action, env.curplayer)] &&
            !env.board[posidx(1, action, !env.curplayer)]
        end
    end

    function full_board(env::BitwiseConnectFourEnv)
        return !any(BatchedEnvs.valid_action(env, action) for action in 1:7)
    end

    function is_win(env::BitwiseConnectFourEnv, player::Bool)
        at(i, j) = env.board[posidx(i, j, player)]
        for i in 1:6, j in 1:7
            if at(i, j) == 1
                if i <= 3 && at(i+1, j) == at(i+2, j) == at(i+3, j) == 1
                    return true
                end
                if j <= 4 && at(i, j+1) == at(i, j+2) == at(i, j+3) == 1
                    return true
                end
                if (i <= 3 && j <= 4) && at(i+1, j+1) == at(i+2, j+2) == at(i+3, j+3) == 1
                    return true
                end
                if (i >= 4 && j <= 4) && at(i-1, j+1) == at(i-2, j+2) == at(i-3, j+3) == 1
                    return true
                end
            end
        end
        return false
    end

    function BatchedEnvs.terminated(env::BitwiseConnectFourEnv)
        return begin
            is_win(env, env.curplayer) ||
            is_win(env, !env.curplayer) ||
            full_board(env)
        end
    end

end