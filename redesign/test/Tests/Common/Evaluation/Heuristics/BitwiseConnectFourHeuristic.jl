module BitwiseConnectFourHeuristic

using ..BitwiseConnectFour

export connect_four_eval_fn

const NUM_COLUMNS = BitwiseConnectFour.NUM_COLUMNS
const NUM_ROWS = BitwiseConnectFour.NUM_ROWS
const TO_CONNECT = BitwiseConnectFour.TO_CONNECT

valid_pos((col, row)) = 1 <= col <= NUM_COLUMNS && 1 <= row <= NUM_ROWS
posidx(n, player) = n + (NUM_ROWS * NUM_COLUMNS) * player
posidx(x, y, player) = posidx(NUM_COLUMNS * (x - 1) + y, player)
at(env, i, j, player) = env.board[posidx(i, j, player)]

function alignment_from(pos, direction)
    al = Vector{Tuple{Int, Int}}()
    for _ in 1:TO_CONNECT
        valid_pos(pos) || (return nothing)
        push!(al, pos)
        pos = pos .+ direction
    end
    return al
end

function alignments_with(direction)
    als = [alignment_from((x, y), direction) for x in 1:NUM_COLUMNS for y in 1:NUM_ROWS]
    return filter(al -> !isnothing(al), als)
end

const ALIGNMENTS = [
    alignments_with((1,  1));
    alignments_with((1, -1));
    alignments_with((0,  1));
    alignments_with((1,  0))
]

function alignment_value_for(env::BitwiseConnectFourEnv, player, alignment)
    discount = 0.1
    N = 0
    for pos in alignment
        y, x = pos
        if at(env, x, y, player)
            N += 1
        elseif at(env, x, y, !player)
            return 0.
        end
    end
    return discount ^ (TO_CONNECT - 1 - N)
end

function connect_four_eval_fn(env::BitwiseConnectFourEnv, is_x_player::Bool)
    X, O = BitwiseConnectFour.CROSS, BitwiseConnectFour.NOUGHT
    cross_value = sum(alignment_value_for(env, X, al) for al in ALIGNMENTS)
    nought_value = sum(alignment_value_for(env, O, al) for al in ALIGNMENTS)
    # return curr_player_value - opponent_value
    # return is_x_player ? (cross_value - nought_value) : (nought_value - cross_value)
    return cross_value - nought_value
end

end
