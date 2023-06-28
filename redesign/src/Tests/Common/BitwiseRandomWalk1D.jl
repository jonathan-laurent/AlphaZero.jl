module BitwiseRandomWalk1D

using StaticArrays

using ....BatchedEnvs
using ....Util.StaticBitArrays

export BitwiseRandomWalk1DEnv

const ENV_SIZE = 10
const LEFT = 1
const RIGHT = 2

const BitBoard = StaticBitArray{ENV_SIZE, (ENV_SIZE - 1) ÷ 64 + 1}


"""
A random walk environment implemented using bitwise operations
that can be run on GPU.
"""
struct BitwiseRandomWalk1DEnv
    board::BitBoard
    cur_pos::Int8
end

function BitwiseRandomWalk1DEnv()
    start_pos = (ENV_SIZE + 1) ÷ 2
    board = Base.setindex(BitBoard(), true, start_pos)
    return BitwiseRandomWalk1DEnv(board, start_pos)
end

BatchedEnvs.state_size(::BitwiseRandomWalk1DEnv) = ENV_SIZE
BatchedEnvs.num_actions(::BitwiseRandomWalk1DEnv) = 2

function Base.show(io::IO, ::MIME"text/plain", env::BitwiseRandomWalk1DEnv)
    print_pos(pos) = (pos == env.cur_pos) ? print(io, "X ") : print(io, "· ")
    map(print_pos, 1:ENV_SIZE)
end

function BatchedEnvs.act(env::BitwiseRandomWalk1DEnv, action)
    new_pos = (action == LEFT) ? env.cur_pos - 1 : env.cur_pos + 1
    board = Base.setindex(BitBoard(), true, new_pos)
    newenv = BitwiseRandomWalk1DEnv(board, new_pos)
    reward = (new_pos == 1) ? -1.0 : (new_pos == ENV_SIZE) ? 1.0 : 0.0
    return newenv, (; reward=reward, switched=false)
end

function BatchedEnvs.valid_action(::BitwiseRandomWalk1DEnv, action)
    return action == LEFT || action == RIGHT
end

function BatchedEnvs.terminated(env::BitwiseRandomWalk1DEnv)
    return env.cur_pos == 1 || env.cur_pos == ENV_SIZE
end

function BatchedEnvs.vectorize_state(env::BitwiseRandomWalk1DEnv)
    return Float32.(@SVector [env.board[i] for i in 1:ENV_SIZE])
end

end
