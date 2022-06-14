module MctsTests

using Test
using JET
using RLZero
using ReinforcementLearningBase
using ReinforcementLearningEnvironments

using Random: MersenneTwister
using Statistics: mean

using ..TestEnvs

export run_mcts_tests

function uniform_mcts_policy(; n=100)
    return Policy(;
        num_simulations=n,
        num_considered_actions=9,
        value_scale=0.1,
        max_visit_init=50,
        oracle=uniform_oracle,
    )
end

function profile_rollout()
    oracle = RolloutOracle(MersenneTwister(0))
    env = TicTacToeEnv()
    for i in 1:100
        oracle(env)
    end
end

function profile_explore()
    policy = uniform_mcts_policy()
    env = tictactoe_winning()
    rng = MersenneTwister(0)
    return explore(policy, env)
end

function run_mcts_tests()
    @testset "mcts oracle" begin
        @test isapprox(
            [random_walk_value(; N=5, start_pos=i) for i in 2:4], [-0.5, 0, 0.5], atol=0.1
        )
        @test uniform_oracle(RandomWalk1D(; N=5))[2] == 0
    end
    @testset "mcts policy" begin
        policy = uniform_mcts_policy()
        env = tictactoe_winning()
        rng = MersenneTwister(0)
        tree = explore(policy, env)
        best = argmax(completed_qvalues(tree))
        @test legal_action_space(env)[best] == 3
    end
    @testset "mcts inferred" begin
        @test_opt target_modules = (RLZero.MCTS,) profile_rollout()
        #@test_opt target_modules = (RLZero.MCTS,) profile_explore()
    end
end

end
