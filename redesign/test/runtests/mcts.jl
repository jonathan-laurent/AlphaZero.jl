@testset "MCTS oracle" begin
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
