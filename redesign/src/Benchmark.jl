module Benchmark

using StatsBase
using Random: MersenneTwister

using ..BatchedEnvs
using ..BatchedMcts
using ..Util.Devices

include("./Tests/Common/BitwiseTicTacToe.jl")
using .BitwiseTicTacToe

export benchmark_against_MCTS

const DEVICE = CPU()
const TICTACTOE_ACTIONS = 1:9
const N_SAMPLES = 400

"""
    benchmark_against_MCTS(benchmarked_oracle)

Compute a `wins` rate, a `losses` rate & a `draws` rate againt an MCTS oracle.

Some starting TicTacToe positions are first randomly sampled. Then both agents play from
this starting positions. This way garanties a fairness in the benchmark.
Return the `wins`/ `losses`/ `draws` rates as nammed-tuple (; wins, losses, draws).
"""
function benchmark_against_MCTS(benchmarked_oracle)
    rng = MersenneTwister(0)
    samples = [Benchmark.get_random_tictactoe(rng) for _ in 1:N_SAMPLES]
    samples = [sample for sample in samples if !terminated(sample)]

    total = length(samples)

    benchmarked_policy = Policy(; device=DEVICE, oracle=benchmarked_oracle)
    baseline_policy = Policy(; device=DEVICE, oracle=UniformTicTacToeEnvOracle())

    benchmark_rewards = play_duel(deepcopy(samples), benchmarked_policy, baseline_policy)
    baseline_rewards = play_duel(samples, baseline_policy, benchmarked_policy)

    wins = count(benchmark_rewards .== 1) + count(baseline_rewards .== -1)
    losses = count(benchmark_rewards .== -1) + count(baseline_rewards .== 1)
    draws = count(benchmark_rewards .== 0) + count(baseline_rewards .== 0)

    (wins, losses, draws) = (wins, losses, draws) ./ (2 * total)
    return (; wins, losses, draws)
end

function get_random_tictactoe(rng)
    env = BitwiseTicTacToeEnv()
    n_actions = rand(rng, TICTACTOE_ACTIONS)
    actions = sample(rng, TICTACTOE_ACTIONS, n_actions; replace=false)
    return act(env, actions)[1]
end

function play_duel(samples, first_player, second_player)
    rewards = Int16[]
    while !isempty(samples)
        actions = get_best_actions(first_player, samples)
        samples = play_and_save_terminated!(samples, actions, rewards; switch=false)

        isempty(samples) && break
        actions = get_best_actions(second_player, samples)
        samples = play_and_save_terminated!(samples, actions, rewards; switch=true)
    end

    return rewards
end

function get_best_actions(player, samples)
    tree = BatchedMcts.explore(player, samples)
    actions = argmax.(completed_qvalues(tree))

    return actions
end

function play_and_save_terminated!(samples, actions, rewards; switch)
    infos = act.(samples, actions)
    samples = first.(infos)
    samples = filter(!terminated, samples)

    current_rewards = [last(info).reward for info in infos if terminated(first(info))]
    current_rewards = switch ? .-current_rewards : current_rewards

    push!(rewards, current_rewards...)
    return samples
end

end