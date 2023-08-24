module MinimaxTests

using Random
using Test

using ..Common.BitwiseTicTacToe
using ..Common.BitwiseTicTacToeHeuristic
using ..Common.BitwiseConnectFour
using ..Common.BitwiseConnectFourHeuristic
using ...BatchedEnvs
using ...Minimax

export run_minimax_tests

function run_minimax_tests()
    @testset "Deterministic Minimax" test_deterministic_minimax()
    @testset "Stochastic Minimax" test_stochastic_minimax()
end

function test_deterministic_minimax()
    @testset "TicTacToe: Cross player winning" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 2)
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 8)

        a = minimax(env; depth=5, eval_fn=zero_eval_fn, is_maximizing=true)

        # best action should be for cross player to win along main diagonal
        @test a == 9
    end

    @testset "TicTacToe: Nought player winning" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 2)
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 6)
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 9)

        a = minimax(env; depth=5, eval_fn=zero_eval_fn, is_maximizing=false)

        # best action should be for nought player to play bottom left
        @test a == 3
    end

    @testset "TicTacToe: Early pos" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 1)

        a = minimax(env; depth=9, eval_fn=zero_eval_fn, is_maximizing=false)

        # best action should be for nought player to in the middle
        @test a == 5
    end

    @testset "TicTacToe: Cross player winning with small depth + heuristic" begin
        env = BitwiseTicTacToeEnv()
        env, _ = BatchedEnvs.act(env, 5)
        env, _ = BatchedEnvs.act(env, 4)
        env, _ = BatchedEnvs.act(env, 7)
        env, _ = BatchedEnvs.act(env, 3)

        a = minimax(env; depth=2, eval_fn=tictactoe_eval_fn, is_maximizing=true)

        # best action should be for cross player to play on the rightmost column
        @test a in [8, 9]
    end
end

function test_stochastic_minimax()
    @testset "Connect-4: Cross player winning" begin
        env = BitwiseConnectFourEnv()
        env, _ = BatchedEnvs.act(env, 4)
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 3)
        env, _ = BatchedEnvs.act(env, 1)
        env, _ = BatchedEnvs.act(env, 6)
        env, _ = BatchedEnvs.act(env, 1)

        rng = Random.MersenneTwister(0)

        a = stochastic_minimax(env, rng; depth=5, is_maximizing=true, amplify_rewards=true)

        # when rewards are amplified, only the best action should be chosen
        @test a == 5
    end
end

end
