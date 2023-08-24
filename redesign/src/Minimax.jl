module Minimax

using Distributions
using Flux

using ..BatchedEnvs

export zero_eval_fn
export minimax
export stochastic_minimax
export minimax_compute_optimal_actions


zero_eval_fn(_, _) = 0.0
amplify(r) = iszero(r) ? r : Inf * sign(r)

function get_legal_actions(env)
    na = BatchedEnvs.num_actions(typeof(env))
    return [Int16(a) for a in 1:na if BatchedEnvs.valid_action(env, a)]
end

"""
    max_value(env, depth, eval_fn, amplify_rewards, γ)

Applies the maximizing part of the minimax algorithm to find the best action.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached.
In most 2-player environments, we assume by default that the maximizing player
is the first player (usually white/cross).
"""
function max_value(env, depth, eval_fn, amplify_rewards, γ)
    if depth == 0
        val = eval_fn(env, true)
        return (fill(val, BatchedEnvs.num_actions(typeof(env))), val, 0, 0)
    end

    legal_actions = get_legal_actions(env)

    scores = fill(-Inf, BatchedEnvs.num_actions(typeof(env)))
    best_score = -Inf
    best_action = legal_actions[1]
    best_depth = -1

    for action in legal_actions
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0.0
            depth_val = depth
        else
            fn = info.switched ? min_value : max_value
            _, value, _, depth_val = fn(newenv, depth - 1, eval_fn, amplify_rewards, γ)
        end
        reward = amplify_rewards ? amplify(info.reward) : info.reward
        score = γ * value + reward
        scores[action] = score

        if score > best_score || (score == best_score && depth_val > best_depth)
            best_score, best_action, best_depth = score, action, depth_val
        end
    end

    return scores, best_score, best_action, best_depth
end

"""
    min_value(env, depth, eval_fn, amplify_rewards, γ)

Applies the minimizing part of the minimax algorithm to find the best action.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached.
In most 2-player environments, we assume by default that the minimizing player
is the second player (usually black/nought).
"""
function min_value(env, depth, eval_fn, amplify_rewards, γ)
    if depth == 0
        val = eval_fn(env, false)
        return (fill(val, BatchedEnvs.num_actions(typeof(env))), val, 0, 0)
    end

    legal_actions = get_legal_actions(env)

    scores = fill(Inf, BatchedEnvs.num_actions(typeof(env)))
    best_score = Inf
    best_action = legal_actions[1]
    best_depth = -1

    for action in legal_actions
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0.0
            depth_val = depth
        else
            fn = info.switched ? max_value : min_value
            _, value, _, depth_val = fn(newenv, depth - 1, eval_fn, amplify_rewards, γ)
        end
        min_perspective_reward = -info.reward
        reward = amplify_rewards ? amplify(min_perspective_reward) : min_perspective_reward
        score = γ* value + reward
        scores[action] = score

        if score < best_score || (score == best_score && depth_val > best_depth)
            best_score, best_action, best_depth = score, action, depth_val
        end
    end

    return scores, best_score, best_action, best_depth
end

"""
    minimax(
        env;
        depth=5,
        eval_fn=zero_eval_fn,
        is_maximizing=true,
        amplify_rewards=false,
        γ=1.0
    )

Applies the minimax algorithm to find the best action for the current player.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached. If `amplify_rewards` is set to true, environment rewards
will be multipled by a large constant. This can be useful in cases where heuristic
functions might give higher scores than the rewards-to-go. The `γ` parameter is
the discount factor used to compute the action values.

The algorithm works in a deterministic way, by selecting always the action with the
best score achieved in the closest (to the root) possible depth.

The evaluation function is heuristic and should have the following format:

    eval_fn(env, player::Bool) -> Float32

where `env` is an environment following the `BatchedEnvs` interface and `player`
is a boolean indicating whether the current player is the maximizing player.
"""
function minimax(
    env;
    depth=5,
    eval_fn=zero_eval_fn,
    is_maximizing=true,
    amplify_rewards=false,
    γ=1.0
)
    recursive_fn = is_maximizing ? max_value : min_value
    _, _, a, _ = recursive_fn(env, depth, eval_fn, amplify_rewards, γ)
    return a
end

"""
    sample_action(scores, rng)

Samples an action from the scores vector, which is assumed to be a vector containing
a score (q value) for each action. The probabilities of the scores are computed using
a softmax function, and then a sample is drawn the categorical distribution defined
by the probabilities of the scores.
"""
function sample_action(scores, env, rng)
    # if all entries are -Inf, just set the probs to be uniform of valid actions
    if all(isinf.(scores))
        valid_actions = get_legal_actions(env)
        scores[scores .== -Inf] .= 0
        scores[valid_actions] .= 1 / length(valid_actions)
        probs = scores
    # else, just apply softmax to scores
    else
        probs = Flux.softmax(scores)
    end
    return rand(rng, Distributions.Categorical(probs))
end

"""
    stochastic_minimax(
        env,
        rng;
        depth=5,
        eval_fn=zero_eval_fn,
        is_maximizing=true,
        amplify_rewards=false,
        γ=1.0
    )

Applies the minimax algorithm to find the next action for the current player.
Unlike minimax, at the root node the action is not selected deterministically
(argmax/argmin), but rather it samples from a categorical distribution of the scores.
"""
function stochastic_minimax(
    env,
    rng;
    depth=5,
    eval_fn=zero_eval_fn,
    is_maximizing=true,
    amplify_rewards=false,
    γ=1.0
)
    recursive_fn = is_maximizing ? max_value : min_value
    scores, _, _, _ = recursive_fn(env, depth, eval_fn, amplify_rewards, γ)
    (!is_maximizing) && (scores .*= -1)
    return sample_action(scores, env, rng)
end

"""
    max_value_store_optimal_actions!(
        env,
        depth,
        eval_fn,
        optimal_policy_dict,
        possible_envs
    )

Applies the maximizing part of the minimax algorithm to find the best action, while
storing the optimal policy in a dictionary, as well as new environments encountered.
It returns the best score and the best action.
"""
function max_value_store_optimal_actions!(
    env,
    depth,
    eval_fn,
    optimal_policy_dict,
    possible_envs
)
    (depth == 0) && return (eval_fn(env, true), 0, 0)

    legal_actions = get_legal_actions(env)
    scores = fill(-Inf32, length(legal_actions))

    best_score = -Inf32
    best_action = legal_actions[1]

    for (idx, action) in enumerate(legal_actions)
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0f0
        else
            switch = info.switched
            f = switch ? min_value_store_optimal_actions! : max_value_store_optimal_actions!
            value, _ = f(newenv, depth - 1, eval_fn, optimal_policy_dict, possible_envs)
        end
        score = value + info.reward
        scores[idx] = score

        if score > best_score
            best_score, best_action = score, action
        end
    end

    env_encoding = string(env)
    if !(env_encoding in keys(optimal_policy_dict))
        optimal_actions = []
        for (action, score) in zip(legal_actions, scores)
            (score == best_score) && push!(optimal_actions, action)
        end
        optimal_policy_dict[env_encoding] = optimal_actions
        push!(possible_envs, env)
    end

    return best_score, best_action
end

"""
    min_value_store_optimal_actions!(
        env,
        depth,
        eval_fn,
        optimal_policy_dict,
        possible_envs
    )

Applies the minimizing part of the minimax algorithm to find the best action, while
storing the optimal policy in a dictionary, as well as new environments encountered.
It returns the best score and the best action.
"""
function min_value_store_optimal_actions!(
    env,
    depth,
    eval_fn,
    optimal_policy_dict,
    possible_envs
)
    (depth == 0) && return (eval_fn(env, false), 0, 0)

    legal_actions = get_legal_actions(env)
    scores = fill(Inf32, length(legal_actions))

    best_score = Inf32
    best_action = legal_actions[1]

    for (idx, action) in enumerate(legal_actions)
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0f0
        else
            switch = info.switched
            f = switch ? max_value_store_optimal_actions! : min_value_store_optimal_actions!
            value, _ = f(newenv, depth - 1, eval_fn, optimal_policy_dict, possible_envs)
        end
        min_perspective_reward = -info.reward
        score = value + min_perspective_reward
        scores[idx] = score

        if score < best_score
            best_score, best_action = score, action
        end
    end

    env_encoding = string(env)
    if !(env_encoding in keys(optimal_policy_dict))
        optimal_actions = []
        for (action, score) in zip(legal_actions, scores)
            (score == best_score) && push!(optimal_actions, action)
        end
        optimal_policy_dict[env_encoding] = optimal_actions
        push!(possible_envs, env)
    end

    return best_score, best_action
end

"""
    minimax_compute_optimal_actions(
        env;
        depth=5,
        eval_fn=zero_eval_fn,
        is_maximizing=true
    )

Applies the minimax algorithm to find the best action for the current player, while
storing the optimal policy in a dictionary, as well as new environments encountered.
It returns the best action, the optimal policy dictionary and the list of possible
environments.
"""
function minimax_compute_optimal_actions(
    env;
    depth=5,
    eval_fn=zero_eval_fn,
    is_maximizing=true
)
    optimal_policy_dict = Dict{String, Vector{Int16}}()
    possible_envs = Vector{typeof(env)}()

    fn = is_maximizing ? max_value_store_optimal_actions! : min_value_store_optimal_actions!
    _, a = fn(env, depth, eval_fn, optimal_policy_dict, possible_envs)

    return a, optimal_policy_dict, possible_envs
end


end
