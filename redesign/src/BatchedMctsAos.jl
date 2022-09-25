"""
An batched MCTS implementation that can run on GPU where trees
are represented in Array of Structs format.

See `BatchedMcts` for more information.
Those two implementations have the same interface at one exception:
- The returned states from the `EnvOracle` used in the `Policy` must be `isbits` if we want
to run the MCTS on GPU. It is therefore impossible to return a dimensional `CuArray` as a
state representation.
"""
module BatchedMctsAos

using Flux
using StaticArrays
using Distributions: Gumbel
using Random: AbstractRNG
using Base: @kwdef
using Setfield
using Base.Iterators: map as imap
using EllipsisNotation
using CUDA: @allowscalar

using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax
using ..BatchedMctsUtility

export explore, gumbel_explore, completed_qvalues

## Value stored in `tree.parent` for nodes with no parents.
const NO_PARENT = Int16(0)
## Value stored in `tree.children` for unvisited children.
const UNVISITED = Int16(0)
## Value used in various tree's attributes to access root information.
const ROOT = Int16(1)
## Valud used when no action is selected.
const NO_ACTION = Int16(0)

@kwdef struct Node{NumActions,State}
    # Static info created at initialization
    state::State
    parent::Int16 = NO_PARENT
    prev_action::Int16 = NO_ACTION
    prev_reward::Float32 = 0.0f0
    prev_switched::Bool = false
    terminal::Bool = false
    valid_actions::SVector{NumActions,Bool} = @SVector fill(false, NumActions)
    # Oracle info
    prior::SVector{NumActions,Float32} = @SVector zeros(Float32, NumActions)
    oracle_value::Float32 = 0.0f0
    # Dynamic info
    children::SVector{NumActions,Int16} = @SVector zeros(Int16, NumActions)
    num_visits::Int16 = UNVISITED
    total_rewards::Float32 = 0.0f0
end

l1_normalise(policy) = policy / abs(sum(policy; init=Float32(0)))

function Node{na}(state, valid_actions, prior, oracle_value) where {na}
    prior = l1_normalise(prior .* valid_actions)
    return Node{na,typeof(state)}(; state, valid_actions, prior, oracle_value, num_visits=1)
end

function create_tree(mcts, envs)
    @assert length(envs) != 0 "There should be at least one environment."

    info = mcts.oracle.init_fn(envs)
    na, ne, ns = size(info.policy_prior)[1], length(envs), mcts.num_simulations

    last_state_dims = length(size(info.internal_states))
    internal_states = if last_state_dims == 1
        info.internal_states
    else
        Flux.unstack(info.internal_states, last_state_dims)
    end
    valid_actions = Flux.unstack(info.valid_actions, 2)
    policy_prior = Flux.unstack(info.policy_prior, 2)

    # We index tree nodes with (batchnum, simnum)
    # This is unusual but this has better cache locality in this case
    tree = DeviceArray(mcts.device){Node{na,typeof(info.internal_states[.., 1])}}(
        undef, (ne, ns)
    )
    tree[:, ROOT] =
        Node{na}.(internal_states, valid_actions, policy_prior, info.value_prior)

    return tree
end

const Tree{N,S} = AbstractArray{Node{N,S}}

function tree_dims(tree::Tree{N,S}) where {N,S}
    na = N
    ne, ns = size(tree)
    return (; na, ne, ns)
end

get_state_info(state_example::AbstractArray) = length(state_example), eltype(state_example)
get_state_info(state_example) = 1, typeof(state_example)

function eval_states!(mcts, tree, simnum, parent_frontier)
    (; ne) = tree_dims(tree)

    parent = first
    action = last

    non_terminal_mask = @. action(parent_frontier) != NO_ACTION
    non_terminal_bids = DeviceArray(mcts.device)(@view((1:ne)[non_terminal_mask]))
    # No new node to expand (a.k.a only terminal node on the frontier)
    (length(non_terminal_bids) == 0) && return parent.(parent_frontier)

    # Regroup `action_ids` and `parent_states` for `transition_fn`
    parent_ids = parent.(parent_frontier[non_terminal_bids])
    action_ids = action.(parent_frontier[non_terminal_bids])

    state_example = @allowscalar tree[1, ROOT].state
    state_size, state_type = get_state_info(state_example)
    if (state_size == 1)
        parent_states = map(DeviceArray(mcts.device)(eachindex(parent_ids))) do i
            tree[non_terminal_bids[i], parent_ids[i]].state
        end
    else
        parent_states = DeviceArray(mcts.device){state_type}(
            undef, (state_size, length(action_ids))
        )

        parent_states_ids = Tuple.(CartesianIndices((1:state_size, eachindex(action_ids))))
        Devices.foreach(parent_states_ids, mcts.device) do (state_i, i)
            parent_states[state_i, i] = tree[non_terminal_bids[i], parent_ids[i]].state[state_i]
            return nothing
        end
    end

    info = mcts.oracle.transition_fn(parent_states, action_ids)

    last_state_dims = length(size(info.internal_states))
    states = if last_state_dims == 1
        info.internal_states
    else
        Flux.unstack(info.internal_states, last_state_dims)
    end

    # Create nodes and save `info`
    Devices.foreach(eachindex(action_ids), mcts.device) do i
        (; na) = tree_dims(tree)
        parent_id = parent_ids[i]
        bid = non_terminal_bids[i]
        aid = action_ids[i]

        node = tree[bid, parent_id]
        @set! node.children[aid] = simnum
        tree[bid, parent_id] = node

        state = states[i]
        tree[bid, simnum] = Node{na,typeof(state)}(;
            state,
            parent=parent_id,
            prev_action=aid,
            prev_reward=info.rewards[i],
            prev_switched=info.player_switched[i],
            terminal=info.terminal[i],
            valid_actions=SVector{na}(imap(aid -> info.valid_actions[aid, i], 1:na)),
            prior=SVector{na}(imap(aid -> info.policy_prior[aid, i], 1:na)),
            oracle_value=info.value_prior[i],
        )

        return nothing
    end

    # Update frontier
    frontier = parent.(parent_frontier)
    @inbounds frontier[non_terminal_bids] .= simnum

    return frontier
end

value(node) = node.total_rewards / node.num_visits

qvalue(child) = value(child) * (-1)^child.prev_switched

function root_value_estimate(tree, node, bid)
    total_qvalues = 0.0f0
    total_prior = 0.0f0
    total_visits = UNVISITED
    for (i, cnid) in enumerate(node.children)
        if cnid != UNVISITED
            child = tree[bid, cnid]
            total_qvalues += node.prior[i] * qvalue(child)
            total_prior += node.prior[i]
            total_visits += child.num_visits
        end
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (node.oracle_value + total_visits * children_value) / (1 + total_visits)
end

function completed_qvalues(tree, node, bid)
    root_value = root_value_estimate(tree, node, bid)
    na = length(node.children)
    ret = imap(1:na) do i
        if (!node.valid_actions[i])
            return -Inf32
        end

        cnid = node.children[i]
        return cnid != UNVISITED ? qvalue(tree[bid, cnid]) : root_value
    end
    return SVector{na}(ret)
end

function num_child_visits(tree, node, bid, i)
    cnid = node.children[i]
    return cnid != UNVISITED ? tree[bid, cnid].num_visits : UNVISITED
end

function qcoeff(value_scale, max_visit_init, tree, node, bid)
    na = length(node.children)
    # init is necessary for GPUCompiler right now...
    max_child_visit = maximum(1:na; init=UNVISITED) do i
        num_child_visits(tree, node, bid, i)
    end
    return value_scale * (max_visit_init + max_child_visit)
end

function target_policy(value_scale, max_visit_init, tree, node, bid)
    qs = completed_qvalues(tree, node, bid)
    return log.(node.prior) + qcoeff(value_scale, max_visit_init, tree, node, bid) * qs
end

function select_nonroot_action(value_scale, max_visit_init, tree, node, bid)
    policy = softmax(target_policy(value_scale, max_visit_init, tree, node, bid))
    na = length(node.children)
    total_visits = sum(i -> num_child_visits(tree, node, bid, i), 1:na; init=UNVISITED)
    return Int16(
        argmax(1:na; init=(NO_ACTION, -Inf32)) do i
            ratio = Float32(num_child_visits(tree, node, bid, i)) / (total_visits + 1)
            return policy[i] - ratio
        end,
    )
end

# `cur` is set to one so that selection starts at root node
function select(value_scale, max_visit_init, tree, bid; cur=ROOT)
    (; na) = tree_dims(tree)
    while true
        node = tree[bid, cur]
        if node.terminal
            # returns current terminal, but no action played
            return cur, NO_ACTION
        end
        aid = select_nonroot_action(value_scale, max_visit_init, tree, node, bid)
        @assert aid != NO_ACTION

        cnid = node.children[aid]
        if cnid != UNVISITED
            # The child is already in the tree so we proceed.
            cur = cnid
        else
            # The child is not in the tree so we return its parent along with the action
            # that leads to it.
            return cur, aid
        end
    end
    return nothing
end

# Start from the root and add a new frontier (whose index is returned)
# After the node is added, one expect the oracle to be called on all
# frontier nodes where terminal=false.
function select(mcts, tree)
    (; ne) = tree_dims(tree)

    batch_ids = DeviceArray(mcts.device)(1:ne)
    value_scale, max_visit_init = mcts.value_scale, mcts.max_visit_init
    parent_frontier = map(batch_ids) do bid
        return select(value_scale, max_visit_init, tree, bid)
    end
    return parent_frontier
end

# Value: if terminal node: terminal value / otherwise: network value
function backpropagate!(mcts, tree, frontier)
    (; ne) = tree_dims(tree)
    batch_ids = DeviceArray(mcts.device)(1:ne)
    map(batch_ids) do bid
        sid = frontier[bid]
        node = tree[bid, sid]
        val = node.oracle_value
        while true
            val += node.prev_reward
            (node.prev_switched) && (val = -val)
            @set! node.num_visits += Int16(1)
            @set! node.total_rewards += val
            tree[bid, sid] = node
            if node.parent != NO_PARENT
                sid = node.parent
                node = tree[bid, sid]
            else
                return nothing
            end
        end
    end
    return nothing
end

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; ns) = tree_dims(tree)
    for simnum in 2:ns
        parent_frontier = select(mcts, tree)
        frontier = eval_states!(mcts, tree, simnum, parent_frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

function get_penality(
    num_considered_actions, tree, bid, considered_visits_table, child_visits
)
    (; na) = tree_dims(tree)
    root_node = tree[bid, ROOT]

    num_valid_actions = sum(root_node.valid_actions; init=NO_ACTION)
    num_considered_actions = min(num_considered_actions, num_valid_actions)

    considered_visits = considered_visits_table[num_considered_actions][child_visits]
    penality = imap(1:na) do aid
        if (num_child_visits(tree, root_node, bid, aid) == considered_visits)
            Float32(0)
        else
            -Inf32
        end
    end
    return SVector{na}(penality)
end

function gumbel_select_root_action(
    value_scale,
    max_visit_init,
    num_considered_actions,
    tree,
    bid,
    gumbel,
    considered_visits_table,
    child_visits,
)
    (; na) = tree_dims(tree)
    root_node = tree[bid, ROOT]

    batch_gumbel = SVector{na}(imap(aid -> gumbel[aid, bid], 1:na))
    policy = target_policy(value_scale, max_visit_init, tree, root_node, bid)
    penality = get_penality(
        num_considered_actions, tree, bid, considered_visits_table, child_visits
    )

    scores = batch_gumbel + policy + penality
    return Int16(argmax(scores; init=(NO_ACTION, -Inf32)))
end

function gumbel_select(mcts, tree, simnum, gumbel, considered_visits_table)
    (; ne) = tree_dims(tree)

    value_scale, max_visit_init = mcts.value_scale, mcts.max_visit_init
    num_considered_actions = mcts.num_considered_actions

    parent_frontier = map(DeviceArray(mcts.device)(1:ne)) do bid
        aid = gumbel_select_root_action(
            value_scale,
            max_visit_init,
            num_considered_actions,
            tree,
            bid,
            gumbel,
            considered_visits_table,
            simnum - ROOT,
        )
        @assert aid != NO_ACTION

        cnid = tree[bid, ROOT].children[aid]
        if (cnid != UNVISITED)
            return select(value_scale, max_visit_init, tree, bid; cur=cnid)
        else
            return (ROOT, aid)
        end
    end
    return parent_frontier
end

function gumbel_explore(mcts, envs, rng::AbstractRNG)
    tree = create_tree(mcts, envs)
    (; na, ne, ns) = tree_dims(tree)

    gumbel = DeviceArray(mcts.device)(rand(rng, Gumbel(), (na, ne)))
    considered_visits_table = DeviceArray(mcts.device)(
        get_considered_visits_table(mcts.num_simulations, na)
    )

    for simnum in 2:ns
        parent_frontier = gumbel_select(mcts, tree, simnum, gumbel, considered_visits_table)
        frontier = eval_states!(mcts, tree, simnum, parent_frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

end