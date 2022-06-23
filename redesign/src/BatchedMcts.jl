"""
A batched implementation of MCTS that can run on GPU or CPU.

Differences with BatchedMctsAos:
- The tree is represented as a structure of arrays rather than an array of structures
- We use a different environment interface that is more generic and
  compatible with MuZero (`init_fn` and `transition_fn` instead of
  `BatchedEnv`)
"""
module BatchedMcts

using Adapt
using Base: @kwdef
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

"""
    Policy(; kwds...)

# Keyword Arguments

- `init_fn` takes a batch of states and returns a tuple
   `(; hidden_state, policy_prior, value_prior)` of batched tensors.
- `transition_fn` takes a `states` tensor and an `actions` tensor and
   returns a tuple (; hidden_state, reward, switched, policy_prior, value_prior)
   tensor.
"""
@kwdef struct Policy{Device,InitFn,TransitionFn}
    device::Device
    init_fn::InitFn
    transition_fn::TransitionFn
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

const NO_PARENT = Int16(0)

"""
A batch of MCTS trees.

# Fields

We provide shape information between parentheses: `B` denotes the
batch size, `N` the maximum number of nodes (i.e. number of simulations)
and `A` the number of actions.

## Cached static information

- `state`: state vector or embedding (..., N, B)
- `terminal`: whether a node is a terminal node (N, B)
- `parent`: the id of the parent node or `NO_PARENT` (N, B)
- `valid_actions`: whether or not each action is valid or not (A, N, B)
- `prev_action`: the id of the action leading to this node or 0 (N, B)
- `prev_reward`: the immediate reward obtained when transitioning from
   the parent from the perspective of the parent's player (N, B)
- `prev_switched`: the immediate reward obtained when transitioning from
   the parent from the perspective of the parent's player (N, B)
- `policy_prior`, `value_prior`: as given by the neural network

## Dynamic statistics

- `num_visits`: number of times the node was visited (N, B)
- `total_values`: the sum of all values backpropagated to a node (N, B)
- `children`: node id of all children or 0 for unvisited actions (A, N, B)
"""
struct Tree{
    StateNodeArray,
    BoolNodeArray,
    Int16NodeArray,
    Float32NodeArray,
    BoolActionArray,
    Int16ActionArray,
    Float32ActionArray,
}
    # Cached static info
    state::StateNodeArray
    terminal::BoolNodeArray
    parent::Int16NodeArray
    valid_actions::BoolActionArray
    # Cached transition info
    prev_action::Int16NodeArray
    prev_reward::Float32NodeArray
    prev_switched::BoolNodeArray
    # Oracle info
    policy_prior::Float32ActionArray
    value_prior::Float32NodeArray
    # Dynamic stats
    num_visits::Int16NodeArray
    total_values::Float32NodeArray
    children::Int16ActionArray
end

tree_batch_size(t::Tree) = size(t)[1]

# https://cuda.juliagpu.org/stable/tutorials/custom_structs/
Adapt.@adapt_structure Tree

function select!(mcts, tree, simnum)
    B = tree_batch_size(tree)
    batch_indices = DeviceArray(mcts.device)(1:B)
    frontier = map(batch_indices) do bid
        @assert false, "Not implemented"
    end
    return frontier
end

end
