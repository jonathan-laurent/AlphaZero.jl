module BitwiseConnectFourTests

using Test

using ..BatchedEnvsTests
using ...Common.BitwiseConnectFour
using ....BatchedEnvs
using ....Util.StaticBitArrays

export run_bitwise_connect_four_tests

function run_bitwise_connect_four_tests()
    Env = BitwiseConnectFourEnv
    @testset "bitwise connect-four" begin
        @testset "bitwise batch simulation" test_batch_simulate(Env)
        @testset "GPU friendliness" test_gpu_friendliness(Env; num_actions=7)
        @testset "action handling" test_action_correctness()
        @testset "valid actions" test_valid_actions()
        @testset "terminated" test_terminated()
        @testset "won game" test_is_win()
        @testset "drawn game" test_full_board()
        @testset "state vectorization" test_vectorize_state()
    end
end


"""
After actions [1, 2, 3, 4, 4, 3, 2, 1, 7, 7], the final state should look like this:

- - - - - - -         (1  -  7)  0 0 0 0 0 0 0     0 0 0 0 0 0 0  (43 - 49)
- - - - - - -         (8  - 14)  0 0 0 0 0 0 0     0 0 0 0 0 0 0  (50 - 56)
- - - - - - -  ---->  (15 - 21)  0 0 0 0 0 0 0     0 0 0 0 0 0 0  (57 - 63)
- - - - - - -  ---->  (22 - 28)  0 0 0 0 0 0 0     0 0 0 0 0 0 0  (64 - 70)
O X O X - - O         (29 - 35)  1 0 1 0 0 0 1     0 1 0 1 0 0 0  (71 - 77)
X O X O - - X         (36 - 42)  0 1 0 1 0 0 0     1 0 1 0 0 0 1  (78 - 84)

                                 NOUGHT PLAYER     CROSS PLAYER
"""
function test_action_correctness()
    actions = [1, 2, 3, 4, 4, 3, 2, 1, 7, 7]
    env = BitwiseConnectFourEnv()
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end
    target_state = StaticBitArray{42 * 2, 2}()

    # Nought Player
    target_state = Base.setindex(target_state, true, 29)
    target_state = Base.setindex(target_state, true, 31)
    target_state = Base.setindex(target_state, true, 35)
    target_state = Base.setindex(target_state, true, 37)
    target_state = Base.setindex(target_state, true, 39)

    # Cross Player
    target_state = Base.setindex(target_state, true, 72)
    target_state = Base.setindex(target_state, true, 74)
    target_state = Base.setindex(target_state, true, 78)
    target_state = Base.setindex(target_state, true, 80)
    target_state = Base.setindex(target_state, true, 84)

    @test env.board == target_state
end

function test_valid_actions()
    env = BitwiseConnectFourEnv()
    valid_actions = [action for action in 1:7 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == collect(1:7)

    # X and O fill the first 2 columns
    actions = [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    valid_actions = [action for action in 1:7 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == collect(3:7)
end

function test_terminated()
    env = BitwiseConnectFourEnv()
    @test !BatchedEnvs.terminated(env)

    # X connects-4 along main diagonal
    actions = [1, 2, 2, 3, 3, 4, 3, 4, 4, 6, 4]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test BitwiseConnectFour.is_win(env, BitwiseConnectFour.CROSS)
end

function test_is_win()
    env = BitwiseConnectFourEnv()
    @test !BitwiseConnectFour.is_win(env, BitwiseConnectFour.CROSS)
    @test !BitwiseConnectFour.is_win(env, BitwiseConnectFour.NOUGHT)

    # 0 connects-4 along last row
    actions = [1, 2, 2, 3, 3, 4, 3, 4, 4, 5]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BitwiseConnectFour.is_win(env, BitwiseConnectFour.NOUGHT)
    @test !BitwiseConnectFour.is_win(env, BitwiseConnectFour.CROSS)
end

function test_full_board()
    env = BitwiseConnectFourEnv()
    @test !BitwiseConnectFour.full_board(env)

    """
    These actions result in the following drawn board:

    O O X O X X O
    X X O X O O X
    O O O X X X O
    X X O X X O X
    O O X O O O X
    X O O X X O X
    """
    actions = [
        1, 2, 4, 3, 5, 6, 7,
        1, 3, 2, 7, 4, 1, 5,
        2, 6, 4, 3, 5, 6, 7,
        1, 4, 2, 5, 3, 6, 7,
        1, 3, 2, 5, 4, 6, 7,
        1, 3, 2, 5, 4, 6, 7
    ]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test BitwiseConnectFour.full_board(env)
    @test !BitwiseConnectFour.is_win(env, BitwiseConnectFour.NOUGHT)
    @test !BitwiseConnectFour.is_win(env, BitwiseConnectFour.CROSS)
end

function test_vectorize_state()
    env = BitwiseConnectFourEnv()
    state = BatchedEnvs.vectorize_state(env)

    empty_board = [0.0 for _ in 1:6 * 7]
    expected_state = Float32.(vcat(empty_board, empty_board))

    @test state == expected_state

    # check that the vectorized state is computed correctly after a couple of actions
    env, _ = BatchedEnvs.act(env, 1)  # CROSS played
    env, _ = BatchedEnvs.act(env, 2)  # NOUGHT played
    env, _ = BatchedEnvs.act(env, 3)  # CROSS played
    env, _ = BatchedEnvs.act(env, 2)  # NOUGHT played
    state = BatchedEnvs.vectorize_state(env)

    current_player_offest = 0
    next_player_offset = 6 * 7

    # CROSS to play - the CROSS board goes after the free board and before the NOUGHT board
    expected_state[current_player_offest + 36] = 1.0
    expected_state[next_player_offset + 37] = 1.0
    expected_state[current_player_offest + 38] = 1.0
    expected_state[next_player_offset + 30] = 1.0

    # perform one more action to ensure the change in board order
    env, _ = BatchedEnvs.act(env, 1)  # CROSS played
    state = BatchedEnvs.vectorize_state(env)

    # swap the two player boards in the expected state
    current_player_board = expected_state[next_player_offset + 1:end]
    next_player_board = expected_state[current_player_offest + 1:next_player_offset]
    expected_state = reduce(vcat, [current_player_board, next_player_board])

    # NOUGHT to play - the NOUGHT board goes after the free board and before the CROSS board
    expected_state[next_player_offset + 29] = 1.0

    @test state == expected_state
end

end
