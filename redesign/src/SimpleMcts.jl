"""
A straightforward non-batched implementation of Gumbel MCTS.

# Design choices
- We represent all available actions explicitly for each node.
- We reset the tree everytime (for now).
- All values are from the current player perspective.
- All computations are done using `Float64` but `Float32` is accepted from oracles.
"""
module SimpleMcts

using ReinforcementLearningBase
using Distributions: sample, Gumbel
using Random: AbstractRNG
using Flux: softmax
using Base: @kwdef

export Policy, gumbel_explore, explore, completed_qvalues
export RolloutOracle, uniform_oracle

"""
An MCTS tree.

MCTS trees are represented by graph of structures in memory.
We store Q-values for each nodes instead of of storing values
so as to make it easier to handle terminal states.
"""
mutable struct Tree
    oracle_value::Float32
    children::Vector{Union{Nothing,Tree}}
    prior::Vector{Float32}
    num_visits::Vector{Int32}
    total_rewards::Vector{Float64}
end

"""
An MCTS Policy that leverages an external oracle.
"""
@kwdef struct Policy{Oracle}
    oracle::Oracle
    num_simulations::Int
    num_considered_actions::Int
    value_scale::Float64
    max_visit_init::Int
end

num_children(node::Tree) = length(node.children)

max_child_visits(node::Tree) = maximum(node.num_visits; init=0)

function visited_children_indices(node::Tree)
    return (i for i in eachindex(node.children) if node.num_visits[i] > 0)
end

children_indices(node::Tree) = eachindex(node.children)

function qvalue(node::Tree, i)
    n = node.num_visits[i]
    @assert n > 0
    return node.total_rewards[i] / n
end

function qcoeff(mcts::Policy, node::Tree)
    # The paper introduces a sigma function, which we implement by
    # sigma(q) = qcoeff * q
    return mcts.value_scale * (mcts.max_visit_init + max_child_visits(node))
end

function root_value_estimate(node::Tree)
    total_visits = sum(node.num_visits)
    root_value = node.oracle_value
    visited = collect(visited_children_indices(node))
    if !isempty(visited)
        children_value = sum(node.prior[i] * qvalue(node, i) for i in visited)
        children_value /= sum(node.prior[i] for i in visited)
    else
        children_value = 0.0
    end
    return (root_value + total_visits * children_value) / (1 + total_visits)
end

function completed_qvalues(node::Tree)
    root_value = root_value_estimate(node)
    return map(children_indices(node)) do i
        node.num_visits[i] > 0 ? qvalue(node, i) : root_value
    end
end

function create_node(env::AbstractEnv, oracle)
    prior, value = oracle(env)
    num_actions = length(legal_action_space(env))
    @assert num_actions > 0
    children = convert(Vector{Union{Nothing,Tree}}, fill(nothing, num_actions))
    num_visits = fill(Int32(0), num_actions)
    total_rewards = fill(Float64(0), num_actions)
    return Tree(value, children, prior, num_visits, total_rewards)
end

"""
Run MCTS search with Gumbel exploration noise on the current state
and return an MCTS tree.
"""
function gumbel_explore(mcts::Policy, env::AbstractEnv, rng::AbstractRNG)
    # Creating an empty tree, sampling the Gumbel variables
    # and selecting m actions with top scores.
    node = create_node(env, mcts.oracle)
    gscores = [rand(rng, Gumbel()) for _ in children_indices(node)]
    base_scores = gscores + log.(node.prior)
    num_considered = min(mcts.num_considered_actions, length(node.children))
    @assert num_considered > 0
    considered::Vector = partialsortperm(base_scores, 1:num_considered; rev=true)
    # Sequential halving
    num_prev_sims = 0  # number of performed simulations
    num_halving_steps = Int(ceil(log2(num_considered)))
    sims_per_step = mcts.num_simulations / num_halving_steps
    while true
        num_visits = Int(max(1, floor(sims_per_step / num_considered)))
        for _ in 1:num_visits
            # If we do not have enough simulations left to
            # visit every considered actions, we must visit
            # the most promising ones with higher priority
            if num_prev_sims + num_considered > mcts.num_simulations
                # For the q-values to exist, we need
                # num_simulations > num_conidered_actions
                qs = [qvalue(node, i) for i in considered]
                scores = base_scores[considered] + qcoeff(mcts, node) * qs
                considered = considered[sortperm(scores; rev=true)]
            end
            # We visit all considered actions once
            for i in considered
                run_simulation_from_child(mcts, node, copy(env), i)
                num_prev_sims += 1
                if num_prev_sims >= mcts.num_simulations
                    return node
                end
            end
        end
        # Halving step
        num_considered = max(2, num_considered ÷ 2)
        qs = [qvalue(node, i) for i in considered]
        scores = base_scores[considered] + qcoeff(mcts, node) * qs
        considered = considered[partialsortperm(scores, 1:num_considered; rev=true)]
    end
end

"""
Run MCTS search on the current state and return an MCTS tree.
"""
function explore(mcts::Policy, env::AbstractEnv)
    node = create_node(env, mcts.oracle)
    for _ in 1:(mcts.num_simulations)
        run_simulation(mcts, node, copy(env))
    end
    return node
end

function run_simulation_from_child(mcts::Policy, node::Tree, env::AbstractEnv, i)
    prev_player = current_player(env)
    actions = legal_action_space(env)
    env(actions[i])
    r = reward(env, prev_player)
    switched = prev_player != current_player(env)
    if is_terminated(env)
        next_value = zero(r)
    else
        if isnothing(node.children[i])
            node.children[i] = create_node(env, mcts.oracle)
        end
        child = node.children[i]
        @assert !isnothing(child)
        next_value = run_simulation(mcts, child, env)
    end
    value = r + (switched ? -next_value : next_value)
    node.num_visits[i] += 1
    node.total_rewards[i] += value
    return value
end

function run_simulation(mcts::Policy, node::Tree, env::AbstractEnv)
    i = select_nonroot_action(mcts, node)
    return run_simulation_from_child(mcts, node, env, i)
end

function target_policy(mcts::Policy, node::Tree)
    qs = completed_qvalues(node)
    return softmax(log.(node.prior) + qcoeff(mcts, node) * qs)
end

function select_nonroot_action(mcts::Policy, node::Tree)
    policy = target_policy(mcts, node)
    total_visits = sum(node.num_visits)
    return argmax(
        policy[i] - node.num_visits[i] / (total_visits + 1) for i in 1:length(node.children)
    )
end

#####
## Some standard oracles
#####

"""
Oracle that always returns a value of 0 and a uniform policy.
"""
function uniform_oracle(env::AbstractEnv)
    n = length(legal_action_space(env))
    P = ones(n) ./ n
    V = 0.0
    return P, V
end

"""
Oracle that performs a single random rollout to estimate state value.

Given a state, the oracle selects random actions until a leaf node is reached.
The resulting cumulative reward is treated as a stochastic value estimate.
"""
struct RolloutOracle{RNG<:AbstractRNG}
    rng::RNG
end

function (oracle::RolloutOracle)(env::AbstractEnv)
    env = copy(env)
    rewards = 0.0
    original_player = current_player(env)
    while !is_terminated(env)
        player = current_player(env)
        a = rand(oracle.rng, legal_action_space(env))
        env(a)
        r = reward(env)
        rewards += player == original_player ? r : -r
    end
    n = length(legal_action_space(env))
    P = ones(n) ./ n
    return P, rewards
end

end
