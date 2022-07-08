module BatchedEnvsTests

using ...BatchedEnvs
using Test
using ReinforcementLearningBase
using Random: MersenneTwister
using CUDA

export test_equivalent, test_batch_simulate

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
            env_baseline(action)
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

function test_batch_simulate(Env; N=10)
    test_batch_simulate(Env, Array; N)
    if CUDA.functional()
        test_batch_simulate(Env, CuArray; N)
    end
    @test true
    return nothing
end

end
