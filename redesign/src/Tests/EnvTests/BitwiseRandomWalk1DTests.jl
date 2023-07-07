module BitwiseRandomWalk1DTests

using ReinforcementLearningEnvironments: RandomWalk1D
using Test

using ..BatchedEnvsTests
using ...Common.BitwiseRandomWalk1D
using ....BatchedEnvs
using ....Util.StaticBitArrays

export run_bitwise_random_walk_tests

function run_bitwise_random_walk_tests()
    Env = BitwiseRandomWalk1DEnv
    Target_Env() = RandomWalk1D(; N=10)
    @testset "bitwise random walk 1d" begin
        @testset "RLEnvs equivalence" test_equivalent(Env, Target_Env)
        @testset "bitwise batch simulation" test_batch_simulate(Env)
        @testset "GPU friendliness" test_gpu_friendliness(Env; num_actions=4)
        @testset "action handling" test_action_correctness()
        @testset "valid actions" test_valid_actions()
        @testset "terminated" test_terminated()
        @testset "won episode" test_is_win()
        @testset "state vectorization" test_vectorize_state()
    end
    return nothing
end


"""
After actions [1, 2, 1, 1, 2, 2, 2, 2], the final state should look like this:

· · · · · · X · · ·
"""
function test_action_correctness()
    actions = [1, 2, 1, 1, 2, 2, 2, 2]
    env = BitwiseRandomWalk1DEnv()
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end
    target_state = StaticBitArray{10, 1}()

    initial_state_pos = (10 + 1) ÷ 2

    num_left_actions = count(a -> a == 1, actions)
    num_right_actions = count(a -> a == 2, actions)

    final_state_pos = initial_state_pos + num_right_actions - num_left_actions

    target_state = Base.setindex(target_state, true, final_state_pos)
    @test env.board == target_state
end


function test_valid_actions()
    env = BitwiseRandomWalk1DEnv()
    valid_actions = [action for action in 1:2 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == collect(1:2)

    # perform some actions, verify that the valid actions in the end are correct
    actions = [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1]
    for action in actions
        env, _ = BatchedEnvs.act(env, action)
    end

    valid_actions = [action for action in 1:7 if BatchedEnvs.valid_action(env, action)]
    @test valid_actions == collect(1:2)
end


"""
After actions [1, 1, 1, 1]
final state should look like this

X · · · · · · · · ·
"""
function test_terminated()
    env = BitwiseRandomWalk1DEnv()
    info = nothing
    @test !BatchedEnvs.terminated(env)

    # player reaches left end
    actions = [1, 1, 1, 1]
    for action in actions
        env, info = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test info.reward == -1.0
end


"""
After actions [1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2]
final state should look like this

· · · · · · · · · X
"""
function test_is_win()
    env = BitwiseRandomWalk1DEnv()
    info = nothing

    # player reaches right end
    actions = [1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2]
    for action in actions
        env, info = BatchedEnvs.act(env, action)
    end

    @test BatchedEnvs.terminated(env)
    @test info.reward == 1.0
end

function test_vectorize_state()
    env = BitwiseRandomWalk1DEnv()
    state = BatchedEnvs.vectorize_state(env)

    expected_state = Float32.([0.0 for _ in 1:10])
    expected_state[5] = 1.0

    @test state == expected_state

    # check that the vectorized state is computed correctly after an action
    env, _ = BatchedEnvs.act(env, 1)
    state = BatchedEnvs.vectorize_state(env)

    expected_state = Float32.([0.0 for _ in 1:10])
    expected_state[4] = 1.0

    @test state == expected_state
end

end
