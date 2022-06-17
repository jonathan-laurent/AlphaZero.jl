"""
An batched MCTS implementation that can run on GPU.

Tree: (batch_size * num_simulations) nodes

Ability: use broadcasting for within the kernel.

How do we deal with warps that execute in lockstep?
We must avoid unnecessary branching.
Env simulation is going to desynchronize warps but only momentarily.
"""
module BatchedMCTS

using StaticArrays
using Distributions: sample, Gumbel
using Random: AbstractRNG
using Flux: softmax
using Base: @kwdef

using ..BatchedEnvs

struct Node{State,NumActions}
    state::State
    parent::Int16
    prev_action::Int16
    expanded::Bool
    prior::SVector{NumActions,Float32}
end

function root_node(state, na)
    return Node(state, Int16(0), Int16(0), false, @SVector zeros(Float32, na))
end

"""
For the moment, the oracle is uniform
"""
@kwdef struct Policy{Oracle,ArrayType}
    oracle::Oracle
    mkarray::ArrayType
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float64 = 0.1
    max_visit_init::Int = 50
end

function create_tree(policy, envs)
    env = envs[1]
    na = num_actions(env)
    ne = length(envs)
    ns = policy.num_simulations
    tree = policy.mkarray{Node{typeof(env),na}}(undef, (ns, ne))
    curnode = policy.mkarray{Int16}(undef, ne)
    tree[1, :] = policy.mkarray([root_node(e, na) for e in envs])
    fill!(curnode, 1)
    return tree, curnode
end

function tree_dims(tree::AbstractVector{Node{S,N}}) where {S,N}
    ns, ne = size(tree)
    return (; na=N, ns, ne)
end

function evaluate_states(policy, tree, pos)
    (; na, ns, ne) = tree_dims(tree)
    policy.mkarray(collect(1:ne))
    return nothing
end

end
