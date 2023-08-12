module MinimaxTests

using Test

using ..Common.BitwiseTicTacToe
using ..Common.BitwiseTicTacToeHeuristic
using ...BatchedEnvs
using ...Minimax

export run_minimax_tests

function run_minimax_tests()
    @testset "TicTacToe" test_tictactoe_minimax()
end

function test_tictactoe_minimax()
    @testset "Cross player winning" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 2)
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 8)

        a = deterministic_minimax(env; depth=5, eval_fn=zero_eval_fn, is_maximizing=true)

        # best action should be for cross player to win along main diagonal
        @test a == 9
    end

    @testset "Nought player winning" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 2)
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 6)
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 9)

        a = deterministic_minimax(env; depth=5, eval_fn=zero_eval_fn, is_maximizing=false)

        # best action should be for nought player to play bottom left
        @test a == 3
    end

    @testset "Drawn game" begin
        env = BitwiseTicTacToeEnv()

        a = deterministic_minimax(env; depth=9, eval_fn=zero_eval_fn, is_maximizing=true)

        # best action should be for nought player to play bottom left
        @test a in [1, 3, 5, 7, 9]
    end

    @testset "Cross player winning with small depth + heuristic" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 4)
        env, _ = BatchedEnvs.act(env, 7)
        env, _ = BatchedEnvs.act(env, 3)

        a = deterministic_minimax(env; depth=2, eval_fn=tictactoe_eval_fn,
                                  is_maximizing=true)

        # best action should be for cross player to play on the rightmost column
        @test a in [8, 9]
    end
end

end
