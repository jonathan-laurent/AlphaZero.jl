module Minimax

using ..BatchedEnvs

export zero_eval_fn
export deterministic_minimax
export minimax_compute_optimal_actions


zero_eval_fn(_, _) = 0f0

function get_legal_actions(env)
    na = BatchedEnvs.num_actions(typeof(env))
    return [Int16(a) for a in 1:na if BatchedEnvs.valid_action(env, a)]
end


"""
    deterministic_max_value(env, depth, eval_fn)

Applies the maximizing part of the minimax algorithm to find the best action.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached.
In most 2-player environments, we assume by default that the maximizing player
is the first player (usually white/cross).
"""
function deterministic_max_value(env, depth, eval_fn)
    (depth == 0) && return (eval_fn(env, true), 0, 0)

    legal_actions = get_legal_actions(env)

    best_score = -Inf32
    best_action = legal_actions[1]
    best_depth = -1

    for action in legal_actions
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0f0
            score_depth = depth
        else
            recursive_fn = info.switched ? deterministic_min_value : deterministic_max_value
            value, _, score_depth = recursive_fn(newenv, depth - 1, eval_fn)
        end
        score = value + info.reward

        if score > best_score || (score == best_score && score_depth > best_depth)
            best_score, best_action, best_depth = score, action, score_depth
        end
    end

    return best_score, best_action, best_depth
end


"""
    deterministic_min_value(env, depth, eval_fn)

Applies the minimizing part of the minimax algorithm to find the best action.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached.
In most 2-player environments, we assume by default that the minimizing player
is the second player (usually black/nought).
"""
function deterministic_min_value(env, depth, eval_fn)
    (depth == 0) && return (eval_fn(env, false), 0, 0)

    legal_actions = get_legal_actions(env)

    best_score = Inf32
    best_action = legal_actions[1]
    best_depth = -1

    for action in legal_actions
        newenv, info = BatchedEnvs.act(env, action)

        if BatchedEnvs.terminated(newenv)
            value = 0f0
            score_depth = depth
        else
            recursive_fn = info.switched ? deterministic_max_value : deterministic_min_value
            value, _, score_depth = recursive_fn(newenv, depth - 1, eval_fn)
        end
        min_perspective_reward = -info.reward
        score = value + min_perspective_reward

        if score < best_score || (score == best_score && score_depth > best_depth)
            best_score, best_action, best_depth = score, action, score_depth
        end
    end

    return best_score, best_action, best_depth
end


"""
    deterministic_minimax(env, depth=5, eval_fn=zero_eval_fn; is_maximizing=true)

Applies the minimax algorithm to find the best action for the current player.
It accepts the current environment `env`, the current depth `depth` and an
evaluation function `eval_fn` that evaluates the current state of the game
when 0 depth is reached. The algorithm works in a deterministic way, by selecting
always the action with the best score achieved in the closest (ro the root) possible depth.

The evaluation function is heuristic and should have the following format:

    eval_fn(env, player::Bool) -> Float32

where `env` is an environment following the `BatchedEnvs` interface and `player`
is a boolean indicating whether the current player is the maximizing player.
"""
function deterministic_minimax(env; depth=5, eval_fn=zero_eval_fn, is_maximizing=true)
    fn = is_maximizing ? deterministic_max_value : deterministic_min_value
    _, a, _ = fn(env, depth, eval_fn)
    return a
end




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
