"""
An batched MCTS implementation that can run on GPU where trees
are represented in Array of Structs format.
"""
module BatchedMctsAos

using StaticArrays
using Distributions: sample, Gumbel
using Random: AbstractRNG
using Base: @kwdef
using Setfield
using Base.Iterators: map as imap

using ..BatchedEnvs
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

@kwdef struct Policy{Oracle,Device}
    oracle::Oracle
    device::Device
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

@kwdef struct Node{NumActions,State}
    # Static info created at initialization
    state::State
    parent::Int16 = Int16(0)
    prev_action::Int16 = Int16(0)
    prev_reward::Float32 = 0.0f0
    prev_switched::Bool = false
    terminal::Bool = false
    valid_actions::SVector{NumActions,Bool} = @SVector zeros(Bool, NumActions)
    # Oracle info
    prior::SVector{NumActions,Float32} = @SVector zeros(Float32, NumActions)
    value_prior::Float32 = 0.0f0
    # Dynamic info
    children::SVector{NumActions,Int16} = @SVector zeros(Int16, NumActions)
    total_rewards::Float32 = 0.0f0
    num_visits::Int16 = Int16(1)
end

function Node{na}(state; args...) where {na}
    terminal = terminated(state)
    if terminal
        valid_actions = SVector{na,Bool}(false for _ in 1:na)
    else
        valid_actions = SVector{na,Bool}(false for _ in 1:na)
    end
    return Node{na,typeof(state)}(; state, terminal, valid_actions, args...)
end

function create_tree(mcts, envs)
    env = envs[1]
    na = num_actions(env)
    ne = length(envs)
    ns = mcts.num_simulations
    Arr = DeviceArray(mcts.device)
    # We index tree nodes with (batchnum, simnum)
    # This is unusual but this has better cache locality in this case
    tree = Arr{Node{na,typeof(env)}}(undef, (ne, ns))
    tree[:, 1] = Arr([Node{na}(e) for e in envs])
    return tree
end

const Tree{N,S} = AbstractArray{Node{N,S}}

function tree_dims(tree::Tree{N,S}) where {N,S}
    na = N
    ne, ns = size(tree)
    return (; na, ne, ns)
end

function eval_states!(mcts, tree, frontier)
    (; na, ne) = tree_dims(tree)
    prior = (@SVector ones(Float32, na)) / na
    Devices.foreach(1:ne, mcts.device) do batchnum
        nid = frontier[batchnum]
        node = tree[batchnum, nid]
        if !node.terminal
            @set! node.prior = prior
            @set! node.value_prior = 0.0f0
            tree[batchnum, nid] = node
        end
    end
    return nothing
end

value(node) = node.total_rewards / node.num_visits

qvalue(child) = value(child) * (-1)^child.prev_switched

function root_value_estimate(tree, node, bid)
    total_qvalues = 0.0f0
    total_prior = 0.0f0
    total_visits = 0
    for (i, cnid) in enumerate(node.children)
        if cnid > 0  # if the child was visited
            child = tree[bid, cnid]
            total_qvalues += node.prior[i] * qvalue(child)
            total_prior += node.prior[i]
            total_visits += child.num_visits
        end
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (node.value_prior + total_visits * children_value) / (1 + total_visits)
end

function completed_qvalues(tree, node, bid)
    root_value = root_value_estimate(tree, node, bid)
    na = length(node.children)
    ret = imap(1:na) do i
        cnid = node.children[i]
        return cnid > 0 ? value(tree[bid, cnid]) : root_value
    end
    return SVector{na}(ret)
end

function num_child_visits(tree, node, bid, i)
    cnid = node.children[i]
    return cnid > 0 ? tree[bid, cnid].num_visits : Int16(0)
end

function qcoeff(mcts, tree, node, bid)
    na = length(node.children)
    # init is necessary for GPUCompiler right now...
    max_child_visit = maximum(1:na; init=Int16(0)) do i
        num_child_visits(tree, node, bid, i)
    end
    return mcts.value_scale * (mcts.max_visit_init + max_child_visit)
end

function target_policy(mcts, tree, node, bid)
    qs = completed_qvalues(tree, node, bid)
    return softmax(log.(node.prior) + qcoeff(mcts, tree, node, bid) * qs)
end

function select_nonroot_action(mcts, tree, node, bid)
    policy = target_policy(mcts, tree, node, bid)
    na = length(node.children)
    total_visits = sum(i -> num_child_visits(tree, node, bid, i), 1:na; init=0)
    return argmax(1:na; init=(0, -Inf32)) do i
        ratio = Float32(num_child_visits(tree, node, bid, i)) / (total_visits + 1)
        return policy[i] - ratio
    end
end

function select!(mcts, tree, simnum, bid)
    (; na) = tree_dims(tree)
    cur = Int16(1)  # start at the root
    while true
        node = tree[bid, cur]
        if node.terminal
            return cur
        end
        i = select_nonroot_action(mcts, tree, node, bid)
        cnid = node.children[i]
        if cnid > 0
            # The child is already in the tree so we proceed.
            cur = cnid
        else
            # The child is not in the tree so we add it and return.
            newstate, info = act(node.state, i)
            child = Node{na}(
                newstate;
                parent=cur,
                prev_action=i,
                prev_reward=info.reward,
                prev_switched=info.switched,
            )
            tree[bid, simnum] = child
            return Int16(simnum)
        end
    end
end

# Start from the root and add a new frontier (whose index is returned)
# After the node is added, one expect the oracle to be called on all
# frontier nodes where terminal=false.
function select!(mcts, tree, simnum)
    (; ne) = tree_dims(tree)
    batch_ids = DeviceArray(mcts.device)(1:ne)
    frontier = map(batch_ids) do bid
        return select!(mcts, tree, simnum, bid)
    end
    return frontier
end

# Value: if terminal node: terminal value / otherwise: network value
function backpropagate!(mcts, tree, frontier)
    (; ne) = tree_dims(tree)
    batch_ids = DeviceArray(mcts.device)(1:ne)
    map(batch_ids) do bid
        node = tree[bid, frontier[bid]]
        val = node.value_prior
        while true
            @set! node.num_visits += Int16(1)
            @set! node.total_rewards += val
            if node.parent > 0
                (node.prev_switched) && (val = -val)
                val += node.prev_reward
                node = tree[bid, node.parent]
            else
                return nothing
            end
        end
    end
    return nothing
end

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; ne, ns) = tree_dims(tree)
    frontier = DeviceArray(mcts.device)(ones(Int16, ne))
    eval_states!(mcts, tree, frontier)
    for i in 2:ns
        frontier = select!(mcts, tree, i)
        eval_states!(mcts, tree, frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

end
