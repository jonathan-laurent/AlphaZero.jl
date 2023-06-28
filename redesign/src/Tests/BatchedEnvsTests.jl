module BatchedEnvsTests

using ...BatchedEnvs
using ....Util.StaticBitArrays

using CUDA
using JET
using Random: MersenneTwister
using ReinforcementLearningBase: RLBase
using Test

export test_equivalent, test_batch_simulate, test_gpu_friendliness

function test_equivalent(BatchedEnv, BaselineEnv)
    rng = MersenneTwister(0)
    for i in 1:100
        env = BatchedEnv()
        env_baseline = BaselineEnv()
        while !BatchedEnvs.terminated(env)
            valid_actions = filter(1:BatchedEnvs.num_actions(env)) do i
                BatchedEnvs.valid_action(env, i)
            end
            valid_actions_baseline = collect(RLBase.legal_action_space(env_baseline))
            @test sort(valid_actions) == sort(valid_actions_baseline)
            @test !isempty(valid_actions)
            action = rand(rng, valid_actions)
            env, info = BatchedEnvs.act(env, action)
            RLBase.act!(env_baseline, action)
            RLBase.next_player!(env_baseline)
            reward_baseline = RLBase.reward(env_baseline)
            info.switched && (reward_baseline *= -1)
            @test info.reward == reward_baseline
        end
        @test RLBase.is_terminated(env_baseline)
    end
    return nothing
end

function find_kth_modulo(f, cands, k)
    n = count(f, cands)
    k = mod(k - 1, n) + 1
    i = 0
    for c in cands
        if f(c)
            i += 1
        end
        if i == k
            return c
        end
    end
    @assert false
end

function test_batch_simulate(Env, device; N=32_000, L=9)
    rng = MersenneTwister(0)
    env = Env()
    envs = device([env for _ in 1:N])
    rd = device(rand(rng, Int, N, L))
    for i in 1:L
        envs = broadcast(envs, rd[:, i]) do e, r
            if !BatchedEnvs.terminated(e)
                a = find_kth_modulo(i -> BatchedEnvs.valid_action(e, i), 1:9, r)
                return BatchedEnvs.act(e, a)[1]
            end
            return e
        end
    end
    return Array(envs)
end

function test_batch_simulate(Env; N=1_000)
    test_batch_simulate(Env, Array; N)
    if CUDA.functional()
        test_batch_simulate(Env, CuArray; N)
    end
    @test true
    return nothing
end

function test_isbits_type(Env)
    env = Env()
    @test isbits(env)
end

function test_is_immutable(Env)
    env = Env()
    @test isimmutable(env)
    for fieldname in fieldnames(typeof(env))
        field_type = typeof(getfield(env, fieldname))
        if field_type <: StaticBitArray
            @test_throws MethodError env.board[1] = false
        elseif field_type == Bool
            @test_throws ErrorException env.curplayer = false
        elseif field_type <: Integer
            @test_throws ErrorException env.curplayer = 1
        end
    end
end

function test_no_allocations(Env, num_actions)
    env = Env()
    rng = MersenneTwister(0)
    actions = rand(rng, 1:num_actions, 5)
    for action in actions
        !BatchedEnvs.valid_action(env, action) && continue
        allocations = @allocated env, _ = BatchedEnvs.act(env, action)
        @test allocations == 0
    end
end

function test_static_inference(Env)
    env = Env()
    @inferred BatchedEnvs.valid_action(env, 1)
    @inferred BatchedEnvs.act(env, 1)
    @inferred BatchedEnvs.terminated(env)

    env = Env()
    @test_opt BatchedEnvs.valid_action(env, 1)
    @test_opt BatchedEnvs.act(env, 1)
    @test_opt BatchedEnvs.terminated(env)
end

function test_gpu_friendliness(Env; num_actions = 7)
    @testset "env type is isbits" test_isbits_type(Env)
    @testset "env is immutable" test_is_immutable(Env)
    @testset "env does not allocate" test_no_allocations(Env, num_actions)
    @testset "env type inference" test_static_inference(Env)
end

end
