module BitwiseTicTacToeTests

using ReinforcementLearningEnvironments: TicTacToeEnv
using Test

using ..BatchedEnvsTests
using ...Common.BitwiseTicTacToe
using ....BatchedEnvs
using ....Util.StaticBitArrays

export run_bitwise_tictactoe_tests

function run_bitwise_tictactoe_tests()
    Env = BitwiseTicTacToeEnv
    @testset "bitwise tictactoe" begin
        @testset "RLEnvs equivalence" test_equivalent(Env, TicTacToeEnv)
        @testset "bitwise batch simulation" test_batch_simulate(Env)
        @testset "GPU friendliness" test_gpu_friendliness(Env; num_actions=5)
        @testset "action handling" test_action_correctness()
        @testset "valid actions" test_valid_actions()
        @testset "terminated" test_terminated()
        @testset "won game" test_is_win()
        @testset "drawn game" test_full_board()
        @testset "state vectorization" test_vectorize_state()
    end
    return nothing
end


"""
After actions [5, 1, 3, 4, 7], the final state should look like this:

O - X             (1 - 3)      1 0 0                    0 0 1       (10 - 12)
O X -  ---->      (4 - 6)      1 0 0                    0 1 0       (13 - 15)
X - -             (7 - 9)      0 0 0                    1 0 0       (16 - 18)

                            NOUGHT PLAYER            CROSS PLAYER
"""
function test_action_correctness()
    actions = [5, 1, 3, 4, 7]
    env = BitwiseTicTacToeEnv()
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end
    target_state = StaticBitArray{9 * 2, 1}()

    # Nought Player
    target_state = Base.setindex(target_state, true, 1)
    target_state = Base.setindex(target_state, true, 4)

    # Cross Player
    target_state = Base.setindex(target_state, true, 12)
    target_state = Base.setindex(target_state, true, 14)
    target_state = Base.setindex(target_state, true, 16)

    @test env.board == target_state
end

function test_valid_actions()
    env = BitwiseTicTacToeEnv()
    valid_actions = [action for action in 1:9 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == collect(1:9)

    # X and O fill the first 2 columns
    actions = [1, 2, 5, 4, 7, 8]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    valid_actions = [action for action in 1:9 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == [3, 6, 9]
end

function test_terminated()
    env = BitwiseTicTacToeEnv()
    @test !BatchedEnvs.terminated(env)

    # X wins along the first column
    actions = [1, 2, 4, 5, 7]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.CROSS)
end

function test_is_win()
    env = BitwiseTicTacToeEnv()
    @test !BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.CROSS)
    @test !BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.NOUGHT)

    # 0 wins along the main diagonal
    actions = [1, 5, 4, 7, 2, 3]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.NOUGHT)
    @test !BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.CROSS)
end

function test_full_board()
    env = BitwiseTicTacToeEnv()
    @test !BitwiseTicTacToe.full_board(env)

    """
    These actions result in the following drawn board:

    O X O
    X O X
    X O X
    """
    actions = [2, 1, 4, 3, 6, 5, 7, 8, 9]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test BitwiseTicTacToe.full_board(env)
    @test !BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.NOUGHT)
    @test !BitwiseTicTacToe.is_win(env, BitwiseTicTacToe.CROSS)
end

function test_vectorize_state()
    env = BitwiseTicTacToeEnv()
    state = BatchedEnvs.vectorize_state(env)

    free_board = [1.0 for _ in 1:3 * 3]
    empty_board = [0.0 for _ in 1:3 * 3]
    expected_state = Float32.(vcat(free_board, empty_board, empty_board))

    @test state == expected_state

    # check that the vectorized state is computed correctly after a couple of actions
    env, _ = BatchedEnvs.act(env, 1)  # CROSS played
    env, _ = BatchedEnvs.act(env, 2)  # NOUGHT played
    env, _ = BatchedEnvs.act(env, 3)  # CROSS played
    env, _ = BatchedEnvs.act(env, 5)  # NOUGHT played
    state = BatchedEnvs.vectorize_state(env)

    current_player_offest = 3 * 3
    next_player_offset = 3 * 3 * 2

    # CROSS to play - the CROSS board goes after the free board and before the NOUGHT board
    expected_state[current_player_offest + 1] = 1.0
    expected_state[1] = 0.0

    expected_state[next_player_offset + 2] = 1.0
    expected_state[2] = 0.0

    expected_state[current_player_offest + 3] = 1.0
    expected_state[3] = 0.0

    expected_state[next_player_offset + 5] = 1.0
    expected_state[5] = 0.0

    # perform one more action to ensure the change in board order
    env, _ = BatchedEnvs.act(env, 6)  # CROSS played
    state = BatchedEnvs.vectorize_state(env)

    # swap the two player boards in the expected state
    free_board = expected_state[1:current_player_offest]
    current_player_board = expected_state[next_player_offset + 1:end]
    next_player_board = expected_state[current_player_offest + 1:next_player_offset]
    expected_state = reduce(vcat, [free_board, current_player_board, next_player_board])

    # NOUGHT to play - the NOUGHT board goes after the free board and before the CROSS board
    expected_state[next_player_offset + 6] = 1.0
    expected_state[6] = 0.0

    @test state == expected_state
end

end
