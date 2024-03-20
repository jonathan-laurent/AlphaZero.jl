module BitwiseTicTacToeHeuristic

using ..BitwiseTicTacToe

export tictactoe_eval_fn


mutable struct TicTacToeStateStatistics
    num_lines_two_x::Int
    num_lines_two_x_zero_o::Int
    num_lines_two_o::Int
    num_lines_two_o_zero_x::Int
end

function tictactoe_eval_fn(env::BitwiseTicTacToeEnv, is_x_player::Bool)
    at(i, j, player) = env.board[BitwiseTicTacToe.posidx(i, j, player)]

    x_player = BitwiseTicTacToe.CROSS
    o_player = BitwiseTicTacToe.NOUGHT

    # heuristic values
    x_controls_center = at(2, 2, x_player) ? 1 : (at(2, 2, o_player) ? -1 : 0)
    o_controls_center = at(2, 2, o_player) ? 1 : (at(2, 2, x_player) ? -1 : 0)
    stats = TicTacToeStateStatistics(0, 0, 0, 0)

    function increment_counters(num_x, num_o, stats)
        (num_x == 2) && (stats.num_lines_two_x += 1)
        (num_x == 2 && num_o == 0) && (stats.num_lines_two_x_zero_o += 1)
        (num_o == 2) && (stats.num_lines_two_o += 1)
        (num_o == 2 && num_x == 0) && (stats.num_lines_two_o_zero_x += 1)
    end

    # rows
    for i in 1:3
        num_crosses = sum(map(j -> at(i, j, x_player), 1:3))
        num_noughts = sum(map(j -> at(i, j, o_player), 1:3))
        increment_counters(num_crosses, num_noughts, stats)
    end

    # columns
    for j in 1:3
        num_crosses = sum(map(i -> at(i, j, x_player), 1:3))
        num_noughts = sum(map(i -> at(i, j, o_player), 1:3))
        increment_counters(num_crosses, num_noughts, stats)
    end

    # main diagonal
    num_crosses = sum(map(i -> at(i, i, x_player), 1:3))
    num_noughts = sum(map(i -> at(i, i, o_player), 1:3))
    increment_counters(num_crosses, num_noughts, stats)

    # second diagonal
    num_crosses = sum(map(c -> at(c[1], c[2], x_player), [(1, 3), (2, 2), (3, 1)]))
    num_noughts = sum(map(c -> at(c[1], c[2], o_player), [(1, 3), (2, 2), (3, 1)]))
    increment_counters(num_crosses, num_noughts, stats)

    # return correct reward if winning in the next move is feasible
    if stats.num_lines_two_x_zero_o > 0 && is_x_player
        return 1.0
    elseif stats.num_lines_two_o_zero_x > 0 && !is_x_player
        return -1.0
    end

    # scale the counts
    x_controls_center *= 0.2
    o_controls_center *= 0.2
    num_lines_two_x = stats.num_lines_two_x * 0.1
    num_lines_two_o = stats.num_lines_two_o * 0.1
    num_lines_two_x_zero_o = stats.num_lines_two_x_zero_o * 0.1
    num_lines_two_x_zero_o = stats.num_lines_two_x_zero_o * 0.1
    turn_factor = is_x_player ? 0.1 : -0.1

    # get subvalues
    x_subvalue = x_controls_center + num_lines_two_x_zero_o + num_lines_two_x
    o_subvalue = o_controls_center + num_lines_two_x_zero_o + num_lines_two_o

    # return the heuristic value
    return turn_factor + x_subvalue - o_subvalue
end

end
