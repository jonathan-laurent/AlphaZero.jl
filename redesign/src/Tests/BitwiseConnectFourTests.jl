module BitwiseConnectFourTests

    using Test
    using ..BatchedEnvsTests
    using ..Common.BitwiseConnectFour
    using ...BatchedEnvs
    using ...Util.StaticBitArrays

    export run_bitwise_connect_four_tests

    function run_bitwise_connect_four_tests()
        @testset "bitwise connect-four" begin
            test_batch_simulate(BitwiseConnectFourEnv)
            test_gpu_friendliness(BitwiseConnectFourEnv; num_actions=7)
            test_action_correctness()
            test_valid_actions()
            test_full_board()
            test_is_win()
            test_terminated()
        end
        return nothing
    end


    """
    After actions [1, 2, 3, 4, 4, 3, 2, 1, 7, 7]
    final state should looks like this

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

end
