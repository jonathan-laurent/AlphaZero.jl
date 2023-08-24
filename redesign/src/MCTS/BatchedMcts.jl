"""
    BatchedMCTS

A high-performance, batched implementation of the Monte Carlo Tree Search (MCTS) algorithm,
suitable for execution on both CPU and GPU architectures.

## Overview

This module provides a generic implementation of MCTS capable of parallelizing simulations
across multiple environment instances. The design is memory-efficient and facilitates a
single, unified representation of all MCTS trees using fixed-size arrays. Data can reside
either in CPU or GPU memory.

The core algorithm is abstracted from any specific environment interface, such as
`ReinforcementLearningBase` or `CommonRLInterface`. Instead, it employs an external
*Environment Oracle* (see `EnvOracle`) for state evaluation and transition simulation,
enabling the use of this MCTS implementation in frameworks like MuZero,
where the environment oracle is often a neural network.

## Algorithm Characteristics and Limitations

- Designed for deterministic, two-player zero-sum games with optional intermediate rewards.
  Single player games are also supported.

- Memory footprint scales with the number of actions available in the environment,
  potentially limiting suitability for environments with a very large or unbounded
  action space.

- Explicit state representation in the search tree optimizes for computational efficiency at
  the expense of increased memory consumption. This is particularly beneficial for MuZero
  where state evaluations are computationally expensive.

## Usage Example

Firstly, initialize a list of environment instances for which optimal actions are to be
determined. Then create an `MctsConfig` object and call an `explore` function:

```jldoctest
julia> envs = [env_constructor() for _ in 1:num_envs]  # Must follow BatchedEnvs interface

julia> device = Devices.GPU()  # can also be `Devices.CPU()`

julia> oracle = EnvOracles.neural_network_env_oracle(; nn=custom_nn)

julia> mcts_config = BatchedMctsUtilities.MctsConfig(; device, oracle, custom_mcts_args...)

julia> tree = BatchedMcts.explore(mcts_config, envs)  # or gumbel_explore, alphazero_explore
```

## Terminology

- bid: Batch identifier, used for indexing the batch (B) dimension of environments.
- cid: Current simulation identifier, used for indexing the simulation (N) dimension.
- cnid: Child identifier, used for indexing the simulation (N) dimension.
- aid: Action identifier, used for indexing the action (A) dimension.
- aids: Vector containing action identifiers.
- qs: Q-values.


# References

- [Policy Improvement by Planning with Gumbel]
(https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel)
"""
module BatchedMcts

using Adapt: @adapt_structure
using Base: @kwdef, size
using CUDA: @inbounds
using Distributions: Dirichlet, Gumbel
using EllipsisNotation
using Random: AbstractRNG
using StaticArrays

import Base.Iterators.map as imap

using ..BatchedMctsUtilities
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax, categorical_sample


export gumbel_explore, explore
export gumbel_policy, evaluation_policy
export get_root_children_visits, get_completed_qvalues


# # Tree datastructure

## Value stored in `tree.parent` for nodes with no parents.
const NO_PARENT = Int16(0)
## Value stored in `tree.children` for unvisited children.
const UNVISITED = Int16(0)
## Value used in various tree's attributes to access root information.
const ROOT = Int16(1)
## Value used when no action is selected.
const NO_ACTION = Int16(0)
# 2 Utilities to easily access information in `parent_frontier`
# `parent_frontier` is a list of tuples with the following format: (parent, action)
const PARENT = Int16(1)
const ACTION = Int16(2)

"""
A data structure encapsulating a batch of Monte Carlo Tree Search (MCTS) trees, utilizing
a Structure of Arrays (SoA) format for efficient computation.

# Fields

The shape of each array field is annotated in parentheses. The symbols `B`, `N`, and `A`
stand for batch size, MCTS simulations, and action space size, respectively.

## Tree Structure and Dynamic Statistics

- `parent` (Int16NodeArray; Shape: [N, B]): Parent node identifiers or a constant
    for root nodes (`NO_PARENT`).
- `num_visits` (Int16NodeArray; Shape: [N, B]): Visit count for each node.
- `total_values` (Float32NodeArray; Shape: [N, B]): Cumulative value estimates for
    each node.
- `children` (Int16ActionArray; Shape: [A, N, B]): Child node identifiers or a constant
    for unexplored actions (`UNVISITED`).

## Cached Static Information from Environment Oracle

These fields are immutable post-population, and they are populated through calls to
an environmental oracle.

- `state` (StateNodeArray; Shape: [.., N, B]): State vectors or embeddings.
- `terminal` (BoolNodeArray; Shape: [N, B]): Flags indicating terminal nodes.
- `valid_actions` (BoolActionArray; Shape: [A, N, B]): Boolean flags indicating the
    validity of each action.
- `prev_action` (Int16NodeArray; Shape: [N, B]): Action identifiers that led to each node.
- `prev_reward` (Float32NodeArray; Shape: [N, B]): Immediate rewards resulting from the
    parent node transitions.
- `prev_switched` (BoolNodeArray; Shape: [N, B]): Flags indicating a player switch
    in the previous transition.
- `logit_prior` (Float32ActionArray; Shape: [A, N, B]): Logit priors for each action
    as returned by the environment oracle.
- `policy_prior` (Float32ActionArray; Shape: [A, N, B]): Policy priors for each action.
- `value_prior` (Float32NodeArray; Shape: [N, B]): Initial value estimates for each node.

# Remarks

- The `Tree` struct is type-parametric to allow CPU or GPU memory allocation
    (e.g., using `Array` or `CuArray` types).
- The efficacy of representing batched MCTS trees as a SoA versus an Array of Structures
    (AoS) is still under investigation.
- The optimal dimension order (`N, B` vs. `B, N`) for cache efficiency has
    not been definitively established.
- Julia's column-major array storage affects cache locality; thus, the dimension order is
    reversed compared to Python implementations that are row-major.
- Double-linking the tree (through `parent` and `children` fields) facilitates
    efficient backpropagation of `num_visits` and `total_values`.

For additional information on backpropagation algorithms, refer to relevant literature
on the core MCTS algorithm.

## Key for Array Types
- N: Number of MCTS Simulations
- B: Number of Parallel Environments (Batch Size)
- A: Action Space Size
"""
@kwdef struct Tree{
    StateNodeArray,
    BoolNodeArray,
    Int16NodeArray,
    Float32NodeArray,
    BoolActionArray,
    Int16ActionArray,
    Float32ActionArray,
}
    ## Dynamic stats
    parent::Int16NodeArray                 # (N, B)      {Int16}
    num_visits::Int16NodeArray             # (N, B)      {Int16}
    total_values::Float32NodeArray         # (N, B)      {Float32}
    children::Int16ActionArray             # (A, N, B)   {Int16}
    ## Cached oracle info
    state::StateNodeArray                  # (N, B)      {BitwiseEnv}
    terminal::BoolNodeArray                # (N, B)      {Bool}
    valid_actions::BoolActionArray         # (A, N, B)   {Bool}
    prev_action::Int16NodeArray            # (N, B)      {Int16}
    prev_reward::Float32NodeArray          # (N, B)      {Float32}
    prev_switched::BoolNodeArray           # (N, B)      {Bool}
    logit_prior::Float32ActionArray        # (A, N, B)   {Float32}
    policy_prior::Float32ActionArray       # (A, N, B)   {Float32}
    value_prior::Float32NodeArray          # (N, B)      {Float32}
end

## https://cuda.juliagpu.org/stable/tutorials/custom_structs/
@adapt_structure Tree


"""
    dims(arr::AbstractArray)
    dims(_)

Return the dimensions of an object, facilitating compatibility within `create_tree`.

This utility function is designed to handle both array and non-array objects.
- For arrays, it returns their dimensions, equivalent to `size(arr)`.
- For non-array objects, it returns an empty tuple, thereby avoiding errors that the
    native `size` function would raise.
"""
dims(arr::AbstractArray) = size(arr)
dims(_) = ()

"""
    create_tree(mcts, envs)

Construct an MCTS `Tree` instance with initial values populated based on the
given environments.

The root node of the tree is considered as explored; it is initialized through a
call to the `init_fn()` of the environmental oracle.

Parameters:
- `mcts`: An MctsConfig instance containing the environmental oracle and
    simulation settings.
- `envs`: A batch of environments for which the tree is constructed.

The function performs multiple initializations, including visit counts, internal states,
and priors, based on the initial information provided by the oracle.

Refer to the [`Tree`](@ref) struct for further details on the data structure.

# Returns
- A `Tree` instance with fields populated according to the initial state of
    the environments and the MCTS algorithm settings.

"""
function create_tree(mcts, envs)
    @assert length(envs) != 0 "There should be at least one environment"

    info = mcts.oracle.init_fn(envs)
    A, N, B = size(info.policy_prior)[1], mcts.num_simulations, length(envs)

    num_visits = fill(UNVISITED, mcts.device, (N, B))
    num_visits[ROOT, :] .= 1

    internal_states = DeviceArray(mcts.device){eltype(info.internal_states)}(
        undef, (dims(info.internal_states[.., 1])..., N, B)
    )
    internal_states[.., ROOT, :] = info.internal_states

    valid_actions = fill(false, mcts.device, (A, N, B))
    valid_actions[:, ROOT, :] = info.valid_actions

    logit_prior = zeros(Float32, mcts.device, (A, N, B))
    logit_prior[:, ROOT, :] = info.logit_prior

    policy_prior = zeros(Float32, mcts.device, (A, N, B))
    policy_prior[:, ROOT, :] = info.policy_prior

    value_prior = zeros(Float32, mcts.device, (N, B))
    value_prior[ROOT, :] = info.value_prior

    return Tree(;
        parent=fill(NO_PARENT, mcts.device, (N, B)),
        num_visits,
        total_values=zeros(Float32, mcts.device, (N, B)),
        children=fill(UNVISITED, mcts.device, (A, N, B)),
        state=internal_states,
        terminal=fill(false, mcts.device, (N, B)),
        valid_actions,
        prev_action=zeros(Int16, mcts.device, (N, B)),
        prev_reward=zeros(Float32, mcts.device, (N, B)),
        prev_switched=fill(false, mcts.device, (N, B)),
        logit_prior,
        policy_prior,
        value_prior,
    )
end

"""
    add_dirichlet_noise_to_root!(tree, mcts_config, rng, ::Val{A}) where {A}

Utility that adds Dirichlet noise to the root node's policy prior.
"""
function add_dirichlet_noise_to_root!(tree, mcts_config, rng, ::Val{A}) where {A}
    dist = Dirichlet(A, mcts_config.alpha_dirichlet)
    noise = DeviceArray(mcts_config.device)(hcat(rand(rng, dist, (batch_size(tree),))...))
    ϵ = mcts_config.epsilon_dirichlet
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        policy_prior = SVector{A}(imap(aid -> tree.policy_prior[aid, ROOT, bid], 1:A))
        bid_noise = SVector{A}(imap(aid -> noise[aid, bid], 1:A))
        action_mask = SVector{A}(imap(aid -> tree.valid_actions[aid, ROOT, bid], 1:A))
        π′ = (1 - ϵ) .* policy_prior .+ ϵ .* bid_noise
        π′ = π′ .* action_mask
        π′ = π′ ./ sum(π′; init=0f0)
        tree.policy_prior[:, ROOT, bid] .= π′
    end
end

"""
    Base.size(tree::Tree)

Retrieve the dimensions of the MCTS `Tree` instance as a named tuple `(; A, N, B)`.

- `A`: Number of available actions.
- `N`: Number of MCTS simulations.
- `B`: Batch size, i.e., the number of environments.

# Returns
- A named tuple `(; A, N, B)` containing the respective dimensions of the tree.

"""
function Base.size(tree::Tree)
    A, N, B = size(tree.children)
    return (; A, N, B)
end

"""
    batch_size(tree)

Return the number of parallel environments in `tree`.
"""
batch_size(tree) = size(tree).B

"""
    n_actions(tree)

Return the total number of actions in the environments simulated in the `tree`.
"""
n_actions(tree) = size(tree).A

# # MCTS implementation

# ## Basic MCTS functions

raw"""
    value(tree, cid, bid)

Calculate the absolute value of a game position at a given node.

The value is computed as:
\[
\frac{{\text{{prior_value}} + \text{{total_rewards}}}}{\text{{num_visits}}}
\]

- `prior_value`: Estimated value from the oracle.
- `total_rewards`: Cumulative rewards obtained during exploration.
- `num_visits`: Number of visits to the node.

# Returns
- The computed value for the node identified by `cid` in the environment batch `bid`.

See also [`qvalue`](@ref)
"""
value(tree, cid, bid) = tree.total_values[cid, bid] / tree.num_visits[cid, bid]

"""
    qvalue(tree, cid, bid)

Compute the value of a game position relative to its parent node.

The returned value accounts for the perspective of the parent node, flipping the
sign based on whether the turn has switched between the parent and the current node.

# Returns
- The value for the node identified by `cid` in the environment batch `bid`,
adjusted for the parent's perspective.

See also [`value`](@ref)
"""
qvalue(tree, cid, bid) = value(tree, cid, bid) * (-1)^tree.prev_switched[cid, bid]

"""
    root_value_estimate(tree, cid, bid, ::Val{A}) where {A}

Estimate the value of a node based on its children and prior information.

The function combines the node's `value_prior`, number of visits,
the `qvalue` of its children, and their `policy_prior` to compute an estimate for the
value of the root node.

# Returns
- The estimated value for the node identified by `cid` in the environment batch `bid`.
"""
function root_value_estimate(tree, cid, bid, ::Val{A}) where {A}
    total_qvalues = Float32(0)
    total_prior = Float32(0)
    total_visits = UNVISITED
    for aid in 1:A
        cnid = tree.children[aid, cid, bid]
        (cnid == UNVISITED) && continue

        total_qvalues += tree.policy_prior[aid, cid, bid] * qvalue(tree, cnid, bid)
        total_prior += tree.policy_prior[aid, cid, bid]
        total_visits += tree.num_visits[cnid, bid]
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (tree.value_prior[cid, bid] + total_visits * children_value) / (1 + total_visits)
end

"""
    completed_qvalues(
        tree,
        cid,
        bid,
        num_actions::Val{A};
        invalid_actions_value=-Inf32
    ) where {A}

Estimate qvalues for all children of a node.

For each child, if visited, it uses its actual `qvalue`. Else, it uses the node's
`root_value_estimate`.

# Returns
- An SVector of size (num_actions,) containing the estimated qvalues for each action.
"""
function completed_qvalues(
    tree,
    cid,
    bid,
    num_actions::Val{A};
    invalid_actions_value = -Inf32
) where {A}
    root_value = root_value_estimate(tree, cid, bid, num_actions)
    ret = imap(1:A) do aid
        (!tree.valid_actions[aid, cid, bid]) && return invalid_actions_value

        cnid = tree.children[aid, cid, bid]
        return cnid != UNVISITED ? qvalue(tree, cnid, bid) : root_value
    end
    return SVector{A}(ret)
end

"""
    alphazero_qvalues(tree, cid, bid, ::Val{A}) where {A}

Compute qvalues for all children of a node.

Unlike in Gumbel MCTS, the qvalues of unvisited children are estimated as zero.

# Returns
- An SVector of size (num_actions,) containing the qvalues for each action.
"""
function alphazero_qvalues(tree, cid, bid, ::Val{A}) where {A}
    ret = imap(1:A) do aid
        cnid = tree.children[aid, cid, bid]
        (cnid != UNVISITED) ? qvalue(tree, cnid, bid) : 0f0
    end
    return SVector{A}(ret)
end

raw"""
    get_puct_values(c_puct, tree, cid, bid, num_actions::Val{A}) where {A}

Compute the PUCT scores for all children of a node.

The PUCT score is computed as:
\[
c_{puct} \text{{policy_prior}} \frac{\sqrt{\text{{total_visits}}}}{1 + \text{{num_visits}}}
\]

# Returns
- An SVector of size (num_actions,) containing the PUCT scores for each action.
"""
function get_puct_values(c_puct, tree, cid, bid, num_actions::Val{A}) where {A}
    # get the policy prior P(s, a)
    policy_prior = SVector{A}(imap(aid -> tree.policy_prior[aid, cid, bid], 1:A))
    # get the visit counts N(s, a) and the total visits sum_b N(s, b) = N(s)
    num_child_visits = Float32.(get_num_child_visits(tree, cid, bid, num_actions))
    total_visits = sum(num_child_visits; init=UNVISITED)

    # compute the scores: c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    scores = c_puct * policy_prior .* (sqrt(total_visits) ./ (1 .+ num_child_visits))

    return scores
end

"""
    get_num_child_visits(tree, cid, bid, ::Val{A}) where {A}

Returns an SVector of visit counts for each child of a given node.

# Returns
- An SVector of size (num_actions,) containing the number of visits for each child action.
"""
function get_num_child_visits(tree, cid, bid, ::Val{A}) where {A}
    ret = imap(1:A) do aid
        cnid = tree.children[aid, cid, bid]
        (cnid != UNVISITED) ? tree.num_visits[cnid, bid] : UNVISITED
    end
    return SVector{A}(ret)
end

"""
    qcoeff(value_scale, max_visit_init, tree, cid, bid, num_actions)

Computes a Gumbel-related weight for the qvalues based on visit counts and prior parameters.

As visit counts increase over time, the influence of `qvalue` relative to `policy_prior`
also increases.

# Returns
- A weight to adjust the influence of `qvalue`.
"""
function qcoeff(c_scale, c_visit, tree, cid, bid, num_actions)
    children_visit_counts = get_num_child_visits(tree, cid, bid, num_actions)
    max_child_visit = maximum(children_visit_counts; init=UNVISITED)
    return (c_visit + max_child_visit) * c_scale
end

"""
    transformed_qvalues(
        c_scale,
        c_visit,
        tree,
        cid,
        bid,
        num_actions::Val{A}
    ) where {A}

Computes q-values (action scores), indicating how favorable each action is for a given node.

# Returns
- An SVector of size (num_actions,) containing the transformed qvalues for each action.
"""
function transformed_qvalues(
    c_scale,
    c_visit,
    tree,
    cid,
    bid,
    num_actions::Val{A}
) where {A}
    qvalues = completed_qvalues(tree, cid, bid, num_actions)
    qcoefficient = qcoeff(c_scale, c_visit, tree, cid, bid, num_actions)
    σ_q = qcoefficient * qvalues
    return σ_q
end

"""
    gumbel_select_action(c_scale, c_visit, tree, cid, bid, num_actions::Val{A}) where {A}

Selects the best action for exploration at a given node.

# Parameters
- `c_scale`: Value scale parameter described in the Gumbel MCTS paper.
- `c_visit`: Initial visit count parameter described in the Gumbel MCTS paper.
- `tree`: The MCTS tree.
- `cid`: Node identifier.
- `bid`: Batch identifier.
- `num_actions`: Number of possible actions.

# Returns
- The action to explore if there's at least one valid action; otherwise, `NO_ACTION`.
"""
function gumbel_select_action(c_scale, c_visit, tree, cid, bid, num_actions::Val{A}) where A
    # compute the policy: π′ = softmax(logits + σ(completedQ))
    logits = SVector{A}(imap(aid -> tree.logit_prior[aid, cid, bid], 1:A))
    σ_q = transformed_qvalues(c_scale, c_visit, tree, cid, bid, num_actions)
    policy = softmax(logits + σ_q)

    # compute the scores: π′(a) - N(s, a) / (N(s) + 1)
    num_child_visits = Float32.(get_num_child_visits(tree, cid, bid, num_actions))
    total_visits = sum(num_child_visits; init=UNVISITED)
    scores = policy - num_child_visits / (total_visits + 1)

    # mask invalid actions as scores for valid actions could be all negative
    action_mask = SVector{A}(imap(a -> tree.valid_actions[a, cid, bid] ? 0f0 : -Inf32, 1:A))
    scores = scores .+ action_mask

    return Int16(argmax(scores; init=(NO_ACTION, -Inf32)))
end

"""
    alphazero_select_action(c_puct, tree, cid, bid, num_actions::Val{A}) where {A}

Selects the best action for exploration at a given node, following the PUCT formula
presented in the AlphaGoZero paper.

# Parameters
- `c_puct`: Exploration constant described in the AlphaGoZero paper.
- `tree`: The MCTS tree.
- `cid`: Node identifier.
- `bid`: Batch identifier.
- `num_actions`: Number of possible actions.

# Returns
- The action to explore if there's at least one valid action; otherwise, `NO_ACTION`.
"""
function alphazero_select_action(c_puct, tree, cid, bid, num_actions::Val{A}) where {A}
    # get Q-values and U(s, a) (aka PUCT values)
    qvalues = alphazero_qvalues(tree, cid, bid, num_actions)
    puct_values = get_puct_values(c_puct, tree, cid, bid, num_actions)

    # final scores are computed as the sum of qvalues and puct values
    scores = qvalues + puct_values

    # mask invalid actions and select the argmax action
    action_mask = SVector{A}(imap(a -> tree.valid_actions[a, cid, bid] ? 0f0 : -Inf32, 1:A))
    scores = scores .+ action_mask

    return Int16(argmax(scores; init=(NO_ACTION, -Inf32)))
end

"""
    get_considered_visits_table(num_simulations, num_actions)

Returns a table containing the precomputed sequence of visits for each number of considered
actions possible. This table introduces a constraint on the number of visits for each
simulation, and is used to precompute the Gumbel simulations orchestration.
"""
function get_considered_visits_table(num_simulations, num_actions)
    ret = imap(1:num_actions) do num_considered_actions
        get_considered_visits_sequence(num_considered_actions, num_simulations)
    end
    return SVector{num_actions}(ret)
end

"""
    get_considered_visits_sequence(max_num_actions, num_simulations)

Precomputes the Gumbel simulations orchestration by applying the Sequential Halving
algorithm iteratively. The output is a sequence of the considered number of visits for
each simulation.
"""
function get_considered_visits_sequence(max_num_actions, num_simulations)
    max_num_actions <= 1 && return SVector{num_simulations, Int16}(0:(num_simulations - 1))

    num_halving_steps = Int(ceil(log2(max_num_actions)))
    sequence = Int16[]
    visits = zeros(Int16, max_num_actions)

    num_actions = max_num_actions
    while length(sequence) < num_simulations
        num_extra_visits = max(1, num_simulations ÷ (num_halving_steps * num_actions))
        for _ in 1:num_extra_visits
            append!(sequence, visits[1:num_actions])
            visits[1:num_actions] .+= 1
        end
        num_actions = max(2, num_actions ÷ 2)
    end

    return SVector{num_simulations}(sequence[1:num_simulations])
end

# ## Core MCTS algorithm

"""
# BatchedMcts Exploration Algorithms

This module provides a batched implementation of Monte Carlo Tree Search (MCTS)
featuring three exploration strategies:
- `explore()`
- `gumbel_explore()`
- `alphazero_explore()`

These algorithms are designed for different contexts and offer various trade-offs between
exploration and exploitation.

## Core MCTS Algorithm

"MCTS" stands for "Monte Carlo Tree Search," a heuristic tree search algorithm primarily
applied in the context of board games and reinforcement learning. The algorithm focuses on
finding the optimal policy — i.e., the best action given a specific state—by exploring a
subset of the total search space. Over time, MCTS has been shown to approximate
the MiniMax algorithm but at a reduced computational cost.

The original MCTS algorithm involved random sampling of the search space,
known as "rollouts," to evaluate states. Modern variations often replace
rollouts with neural network evaluations for increased accuracy.

MCTS iteratively runs simulations to expand a tree of explored states. Each iteration
consists of three main phases:
- **Selection**: Traverse from the root to a leaf node based on a heuristic (e.g., UCB1),
    considering both exploration and exploitation.
- **Evaluation**: Evaluate the leaf node using either rollouts or a neural network and
    expand it to include child nodes corresponding to possible actions.
- **Backpropagation**: Update the visit count and value estimates of all nodes along
    the traversed path based on the evaluation of the leaf node.

## Exploration Strategies

### `explore()`
Suited for inference contexts, this function aims to find the optimal policy without
added exploration noise.

### `gumbel_explore()`
Designed for training contexts in AlphaZero/MuZero frameworks, this function introduces
Gumbel noise during the selection phase to encourage exploration of slightly sub-optimal
actions.

### `alphazero_explore()`
This is the original MCTS algorithm as described in the AlphaGoZero paper. While it might
be outperformed by newer algorithms, it serves as a solid baseline for comparisons.

## Usage

All *explore()* functions return a tree structure. See the main `BatchedMcts` description
for an example and usage instructions, including how to specify different configurations
using `MctsConfig`.

## References
- [DeepMind's MCTX](https://github.com/deepmind/mctx/tree/main/mctx/_src)
- [Policy Improvement by Planning with Gumbel]
(https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel)

"""
function gumbel_explore(mcts_config, envs, rng::AbstractRNG)
    tree = create_tree(mcts_config, envs)
    (; A, B, N) = size(tree)
    gumbel = DeviceArray(mcts_config.device)(rand(rng, Gumbel(), A, B))
    considered_visits = DeviceArray(mcts_config.device)(get_considered_visits_table(N, A))
    for t in 2:N
        parent_frontier = gumbel_select(mcts_config, tree, t, gumbel, considered_visits)
        frontier = eval!(mcts_config, tree, t, parent_frontier)
        backpropagate!(mcts_config, tree, frontier)
    end
    return tree, gumbel
end

function alphazero_explore(mcts_config, envs, rng::AbstractRNG)
    tree = create_tree(mcts_config, envs)
    (; A, N) = size(tree)
    add_dirichlet_noise_to_root!(tree, mcts_config, rng, Val(A))
    for t in 2:N
        parent_frontier = search(mcts_config, tree)
        frontier = eval!(mcts_config, tree, t, parent_frontier)
        backpropagate!(mcts_config, tree, frontier)
    end
    return tree
end

function explore(mcts_config, envs)
    tree = create_tree(mcts_config, envs)
    (; N) = size(tree)
    for t in 2:N
        parent_frontier = search(mcts_config, tree)
        frontier = eval!(mcts_config, tree, t, parent_frontier)
        backpropagate!(mcts_config, tree, frontier)
    end
    return tree
end

"""
    search(mcts_config, tree)

Returns the parent nodes of the frontier of nodes to expand. It uses either Gumbel MCTS
of the original AlphaZero MCTS algorithm, depending on the type of `mcts_config`.

### Note
This implementation leverages Julia's CUDA API to allow the same code to run on both
CPU and GPU. For instance, functions mapped over a CUDA `CuArray` will be implicitly
compiled and run on the GPU, while the same functions mapped over a standard CPU `Array`
will run on the CPU.

### GPU Constraints
While enabling significant performance gains, GPU compilation imposes certain constraints:
- Data must be `isbits` types.
- No dynamic memory allocation on the GPU.
- Functions compiled to GPU must be type-stable.

To address these constraints, the `SVector` type from the `StaticArrays` package is used in
GPU functions and pass `num_actions` as Value-as-parameter for type-stability.

### Parallelization
The batch processing is parallelized over a vector of environments. The `search()` and
`backpropagate!()` functions launch GPU kernels over this vector of environments,
enabling efficient batch processing.

## References
- [DeepMind's MCTX](https://github.com/deepmind/mctx/tree/main/mctx/_src)
- [Policy Improvement by Planning with Gumbel]
(https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel)
"""
function search(mcts_config, tree)
    num_actions = Val(n_actions(tree))
    use_gumbel = is_gumbel_mcts_config(mcts_config)
    c_scale = use_gumbel ? mcts_config.value_scale : 0f0
    c_visit = use_gumbel ? mcts_config.max_visit_init : 0f0
    c_puct = use_gumbel ? 0f0 : mcts_config.c_puct

    parent_frontier = zeros(Int16, mcts_config.device, (2, batch_size(tree)))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        if use_gumbel
            new_frontier = gumbel_search(c_scale, c_visit, tree, bid, num_actions)
        else
            new_frontier = alphazero_search(c_puct, tree, bid, num_actions)
        end
        @inbounds parent_frontier[PARENT, bid] = new_frontier[PARENT]
        @inbounds parent_frontier[ACTION, bid] = new_frontier[ACTION]
    end

    return parent_frontier
end


"""
    gumbel_search(value_scale, max_visit_init, tree, bid, num_actions; start=ROOT)

Descends the search tree by choosing actions according to the sequential halving
algorithm described in the Gumbel MCTS paper.

# Returns
- A tuple `(node, action)` containing the node and action to explore.
"""
function gumbel_search(value_scale, max_visit_init, tree, bid, num_actions; start=ROOT)
    cur = start
    while true
        if tree.terminal[cur, bid]
            # returns current terminal, but no action played
            return cur, NO_ACTION
        end
        aid = gumbel_select_action(value_scale, max_visit_init, tree, cur, bid, num_actions)
        @assert aid != NO_ACTION

        cnid = tree.children[aid, cur, bid]
        if cnid != UNVISITED
            cur = cnid
        else
            # returns parent and action played
            return cur, aid
        end
    end
    return nothing
end

"""
    alphazero_search(c_puct, tree, bid, num_actions; start=ROOT)

Descends the search tree by choosing actions according to the PUCT formula
described in the original AlphaGoZero paper.

# Returns
- A tuple `(node, action)` containing the node and action to explore.
"""
function alphazero_search(c_puct, tree, bid, num_actions; start=ROOT)
    cur = start
    while true
        if tree.terminal[cur, bid]
            # returns current terminal, but no action played
            return cur, NO_ACTION
        end
        aid = alphazero_select_action(c_puct, tree, cur, bid, num_actions)
        @assert aid != NO_ACTION

        cnid = tree.children[aid, cur, bid]
        if cnid != UNVISITED
            cur = cnid
        else
            # returns parent and action played
            return cur, aid
        end
    end
    return nothing
end

"""
    eval!(mcts_config, tree, simnum, parent_frontier)

Expands the frontier leaf nodes or returns them if they are terminal.

## Note: Evaluation Phase

The evaluation phase of the MCTS algorithm evaluates the current state and simulates
the environment with the `EnvOracle` through the `transition_fn()` interface.
Information related to each action (i.e., `valid_actions` & `policy_prior`)
is saved along with the newly created nodes.

### Note: State Handling
To handle both exact state encodings (like in AlphaZero) and latent space state encodings
(like in MuZero), the EllipsisNotation ("..") is used. It enables handling of states with
any number of dimensions.
"""
function eval!(mcts_config, tree, simnum, parent_frontier)
    B = batch_size(tree)

    # get terminal nodes at `parent_frontier`
    non_terminal_mask = parent_frontier[ACTION, :] .!= NO_ACTION
    non_terminal_bids = DeviceArray(mcts_config.device)(@view((1:B)[non_terminal_mask]))
    # No new node to expand (a.k.a only terminal node on the frontier)
    (length(non_terminal_bids) == 0) && return parent_frontier[PARENT, :]

    # regroup `action_ids` and `parent_states` for `transition_fn()`
    parent_ids = parent_frontier[PARENT, non_terminal_bids]
    action_ids = parent_frontier[ACTION, non_terminal_bids]

    state_cartesian_ids = CartesianIndex.(parent_ids, non_terminal_bids)
    parent_states = tree.state[.., state_cartesian_ids]
    info = mcts_config.oracle.transition_fn(parent_states, action_ids)

    # create nodes and save `info`
    children_cartesian_ids = CartesianIndex.(action_ids, parent_ids, non_terminal_bids)

    @inbounds tree.parent[simnum, non_terminal_bids] = parent_ids
    @inbounds tree.children[children_cartesian_ids] .= simnum
    @inbounds tree.state[.., simnum, non_terminal_bids] = info.internal_states
    @inbounds tree.terminal[simnum, non_terminal_bids] = info.terminal
    @inbounds tree.valid_actions[:, simnum, non_terminal_bids] = info.valid_actions
    @inbounds tree.prev_action[simnum, non_terminal_bids] = action_ids
    @inbounds tree.prev_reward[simnum, non_terminal_bids] = info.rewards
    @inbounds tree.prev_switched[simnum, non_terminal_bids] = info.player_switched
    @inbounds tree.logit_prior[:, simnum, non_terminal_bids] = info.logit_prior
    @inbounds tree.policy_prior[:, simnum, non_terminal_bids] = info.policy_prior
    @inbounds tree.value_prior[simnum, non_terminal_bids] = info.value_prior

    # value priors are from the perspective of child nodes -> negate sign if needed
    Devices.foreach(non_terminal_bids, mcts_config.device) do bid
        @inbounds (tree.prev_switched[simnum, bid]) && (tree.value_prior[simnum, bid] *= -1)
    end

    # Update frontier
    frontier = parent_frontier[PARENT, :]
    @inbounds frontier[non_terminal_bids] .= simnum

    return frontier
end

"""
    backpropagate!(mcts_config, tree, frontier)

Backpropagates the value/reward information up the tree.

## Info: Backpropagation Phase
The backpropagation loop updates the visits count and total value of each node in the tree.
The visits count of each node is incremented by 1, while the total value is updated using
TD learning. The value of the newly created node is estimated by the oracle and summed up
with the cumulative reward up to that point. When the player's turn is switched,
the sign of the TD-learned value is also switched.

Like `search()`, the `backpropagate!()` function is parallelized over the vector of
environments.

"""
function backpropagate!(mcts_config, tree, frontier)
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        @inbounds cid = frontier[bid]
        @inbounds val = tree.value_prior[cid, bid]
        while true
            @inbounds val += tree.prev_reward[cid, bid]
            @inbounds (tree.prev_switched[cid, bid]) && (val = -val)
            @inbounds tree.num_visits[cid, bid] += Int16(1)
            @inbounds tree.total_values[cid, bid] += val
            @inbounds if tree.parent[cid, bid] != NO_PARENT
                @inbounds cid = tree.parent[cid, bid]
            else
                return nothing
            end
        end
    end
end

"""
    gumbel_select(mcts_config, tree, simnum, gumbel, considered_visits_table)

Gumbel's variation of classical `select`.

The only difference lies in the use of `gumbel_select_root_action` to select the action at
the root node. `gumbel_select` then fall back on `select` for non-root node.

See also [`gumbel_select_root_action`](@ref)
"""
function gumbel_select(mcts_config, tree, simnum, gumbel, considered_visits_table)
    num_actions = Val(n_actions(tree))
    c_scale, c_visit = mcts_config.value_scale, mcts_config.max_visit_init
    num_considered_actions = mcts_config.num_considered_actions

    parent_frontier = zeros(Int16, mcts_config.device, 2, batch_size(tree))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        aid = gumbel_select_root_action(
            c_scale,
            c_visit,
            num_considered_actions,
            tree,
            bid,
            gumbel,
            considered_visits_table,
            simnum - ROOT,
            num_actions
        )
        @assert aid != NO_ACTION

        cnid = tree.children[aid, ROOT, bid]
        new_frontier = if (cnid != UNVISITED)
            gumbel_search(c_scale, c_visit, tree, bid, num_actions; start=cnid)
        else
            (ROOT, aid)
        end
        @inbounds parent_frontier[PARENT, bid] = new_frontier[PARENT]
        @inbounds parent_frontier[ACTION, bid] = new_frontier[ACTION]
        return nothing
    end

    return parent_frontier
end

"""
    get_penalty(
        tree,
        bid,
        considered_visits,
        num_actions::Val{A}
    ) where {A}

Returns the penalty that will be applied to ROOT actions depending on their visit count.
Specifically, actions with a visit count different from `considered_visits` will receive
a penalty of -∞, while the others will receive a penalty of 0 (float32).
"""
function get_penalty(
    tree,
    bid,
    considered_visits,
    num_actions::Val{A}
) where {A}
    num_visits = get_num_child_visits(tree, ROOT, bid, num_actions)
    penalty = imap(action -> (num_visits[action] == considered_visits) ? 0f0 : -Inf32, 1:A)
    return SVector{A}(penalty)
end

"""
    compute_penalty(
        num_considered_actions,
        tree,
        bid,
        considered_visits_table,
        target_child_visits,
        num_actions::Val{A}
    ) where {A}

Computes penalty for actions that do not comply with the constraint on the number of visits.

More precisely, if an action does not respect this constraint, it will have a penalty of
`-Inf32` on its computed score before applying the argmax. This ultimately blocks
this action from being selected if at least another action has not been penalized
(which should be the case).

Actions that comply with the constraint on the number of visits have no penalty (i.e. a
penalty of `0`)
"""
function compute_penalty(
    num_considered_actions,
    tree,
    bid,
    considered_visits_table,
    target_child_visits,
    num_actions::Val{A}
) where {A}
    num_valid_actions = sum(aid -> tree.valid_actions[aid, ROOT, bid], 1:A; init=NO_ACTION)
    num_considered_actions = min(num_considered_actions, num_valid_actions)
    considered_visits = considered_visits_table[num_considered_actions][target_child_visits]
    return get_penalty(tree, bid, considered_visits, num_actions)
end

"""
    gumbel_select_root_action(
        c_scale,
        c_visit,
        num_considered_actions,
        tree,
        bid,
        gumbel,
        considered_visits_table,
        target_child_visits,
        num_actions::Val{A}
    ) where {A}

Gumbel's variation of `gumbel_select_action()` for the root node only.

The only difference lies in the use of `gumbel` noise in the computation of the scores
before applying the argmax and the additional constraints on the number of visits.
"""
function gumbel_select_root_action(
    c_scale,
    c_visit,
    num_considered_actions,
    tree,
    bid,
    gumbel,
    considered_visits_table,
    target_child_visits,
    num_actions::Val{A}
) where {A}
    # gumbel random variables: g(a)
    g = SVector{A}(imap(aid -> gumbel[aid, bid], 1:A))

    # get logits and σ(completedQ)
    logits = SVector{A}(imap(aid -> tree.logit_prior[aid, ROOT, bid], 1:A))
    σ_q = transformed_qvalues(c_scale, c_visit, tree, ROOT, bid, num_actions)

    # -∞ penalty to mask out actions not in `argtop(g(a) + logits + σ(completedQ), m)`
    penalty = compute_penalty(num_considered_actions, tree, bid, considered_visits_table,
                              target_child_visits, num_actions)

    # select next action `A_{n+1}` to simulate as the argmax of:
    #   A_{top(m)} = argtop(g(a) + logits + σ(completedQ), m)
    scores = g + logits + σ_q
    masked_scores = scores + penalty
    return Int16(argmax(masked_scores; init=(NO_ACTION, -Inf32)))
end

"""
    gumbel_mcts_action(
        c_scale,
        c_visit,
        tree,
        bid,
        gumbel,
        num_actions::Val{A}
    ) where {A}

A function that returns the action to play in the environment after `gumbel_explore()`
has been run. It's similar to `gumbel_select_root_action()`, with the difference that
it penalizes actions with a visit count smaller than the highest.
"""
function gumbel_mcts_action(
    c_scale,
    c_visit,
    tree,
    bid,
    gumbel,
    num_actions::Val{A}
) where {A}
    # gumbel random variables: g(a)
    g = SVector{A}(imap(aid -> gumbel[aid, bid], 1:A))

    # get logits and σ(completedQ)
    logits = SVector{A}(imap(aid -> tree.logit_prior[aid, ROOT, bid], 1:A))
    σ_q = transformed_qvalues(c_scale, c_visit, tree, ROOT, bid, num_actions)

    # -∞ penalty to mask out actions with a visit count less than the highest
    children_visit_counts = get_num_child_visits(tree, ROOT, bid, num_actions)
    considered_visits = maximum(children_visit_counts; init=UNVISITED)
    penalty = get_penalty(tree, bid, considered_visits, num_actions)

    # select next action `A_{n+1}` to play in the environment
    #   A_{top(m)} = argtop(g(a) + logits + σ(completedQ), m)
    scores = g + logits + σ_q
    masked_scores = scores + penalty
    return Int16(argmax(masked_scores; init=(NO_ACTION, -Inf32)))
end

"""
    gumbel_policy(tree, mcts_config, gumbel)

Returns an array of size (num_envs,) containing the resulting actions selected
by the sequential halving procedure with gumbel for each environment. This function should
be used after `gumbel_explore()` has been run.
"""
function gumbel_policy(tree, mcts_config, gumbel)
    num_actions = Val(n_actions(tree))
    c_scale, c_visit = mcts_config.value_scale, mcts_config.max_visit_init

    actions = zeros(Int16, mcts_config.device, batch_size(tree))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        actions[bid] = gumbel_mcts_action(c_scale, c_visit, tree, bid, gumbel, num_actions)
    end

    return actions
end

"""
    alphazero_policy(tree, mcts_config, rng::AbstractRNG)

Returns an array of size (num_envs,) containing the actions selected by the
AlphaGoZero/AlphaZero action selection algorithm for each environment. This function
should be used after `alphazero_explore()` has been run.
"""
function alphazero_policy(tree, mcts_config, current_steps, rng::AbstractRNG)
    num_actions = Val(n_actions(tree))
    τ = mcts_config.tau
    deterministic_move_idx = mcts_config.collapse_tau_move
    probs = DeviceArray(mcts_config.device)(rand(rng, Float32, batch_size(tree)))

    actions = zeros(Int16, mcts_config.device, batch_size(tree))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        children_visit_counts = get_num_child_visits(tree, ROOT, bid, num_actions)
        if current_steps[bid] >= deterministic_move_idx || τ == 0f0
            actions[bid] = Int16(argmax(children_visit_counts; init=(NO_ACTION, UNVISITED)))
        else
            tau_visits = children_visit_counts .^ (1 / τ)
            tau_total_visits = sum(tau_visits; init=UNVISITED)
            policy = tau_visits / tau_total_visits
            actions[bid] = categorical_sample(policy, probs[bid])
        end
    end

    return actions
end

"""
    evaluation_policy(tree, mcts_config)

Returns an array of size (num_envs,) containing the actions with the highest visit
count for each environment. This function should be used after `explore()` has been run.
"""
function evaluation_policy(tree, mcts_config)
    num_actions = Val(n_actions(tree))

    actions = zeros(Int16, mcts_config.device, batch_size(tree))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        children_visit_counts = get_num_child_visits(tree, ROOT, bid, num_actions)
        actions[bid] = Int16(argmax(children_visit_counts; init=(NO_ACTION, UNVISITED)))
    end

    return actions
end

"""
    get_root_children_visits(tree, mcts_config)

Returns an array of size (num_actions, num_envs) containing the number of visits for
each action at the root node for each environment. This function should be used after
`gumbel_explore()` or `explore()` has been run.
"""
function get_root_children_visits(tree, mcts_config)
    num_actions = Val(n_actions(tree))

    visits = zeros(Int16, mcts_config.device, (n_actions(tree), batch_size(tree)))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        visits[:, bid] .= get_num_child_visits(tree, ROOT, bid, num_actions)
    end

    return visits
end

"""
    get_completed_qvalues(tree, mcts_config)

Returns an array of size (num_actions, num_envs) containing the completed qvalues
for each action at the root node for each environment. This function should be used after
`gumbel_explore()` or `explore()` has been run.
"""
function get_completed_qvalues(tree, mcts_config)
    num_actions = Val(n_actions(tree))

    qvalues = zeros(Float32, mcts_config.device, (n_actions(tree), batch_size(tree)))
    Devices.foreach(1:batch_size(tree), mcts_config.device) do bid
        qvalues[:, bid] .= completed_qvalues(tree, ROOT, bid, num_actions)
    end

    return qvalues
end

end
