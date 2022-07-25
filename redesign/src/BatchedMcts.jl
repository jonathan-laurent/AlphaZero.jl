"""
    BatchedMcts

A batched implementation of MCTS that can run on CPU or GPU.

Because this implementation is batched, it is optimized for running MCTS on a large number
of environment instances in parallel. In particular, this implementation is not suitable for
running a large number of MCTS simulations on a single environment quickly (which would
probably require an asynchronous MCTS implementation with virtual loss).

All MCTS trees are represented with a single structure of fixed-sized arrays. This structure
can be hosted either on CPU memory or GPU memory.

In addition, this implementation is not tied to a particular environment interface (such as
`ReinforcementLearningBase` or `CommonRLInterface`). Instead, it relies on an external
*environment oracle* (see `EnvOracle`) that simulates the environment and evaluates states.
This is particularly useful to implement MuZero, in which case the full environment oracle
is implemented by a neural network (in contrast with AlphaZero where a ground-truth
simulator is available).


# Characteristics and limitations

- This implementation supports deterministic, two-player zero-sum games with or without
  intermediate rewards. Support for one-player games should be easy to add and will probably
  become available in the future.
- The memory footprint of the MCTS tree is proportional to the number of environment actions
  (e.g. 9 for tictactoe and ~19x19 for Go). Therefore, this implementation may not be
  suitable for environments offering a very large (or unbounded) number of actions of which
  only a small subset is available in every state.
- States are represented explicitly in the search tree (in contrast with the `SimpleMcts`
  implementation). This increases the memory footprint but avoids simulating the same
  environment transition multiple times (which is essential for MuZero as doing so requires
  making a costly call to a neural network).


# Usage

TODO: This section should show examples of using the module (using jltest?). Ideally, it
should demonstrate different settings such as:

- An AlphaZero-like setting where everything runs on GPU.
- An AlphaZero-like setting where state evaluation runs on GPU and all the rest runs on CPU
  (tree search and environment simulation).
- A MuZero-like setting where everything runs on GPU.

This section should also demonstrate the use of the `check_policy` function to perform
sanity checks on user environments.


# References

- Reference on the Gumbel MCTS extension:
  https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel
"""
module BatchedMcts

using Adapt: @adapt_structure
using Base: @kwdef, size
using ReinforcementLearningBase

using ..Util.Devices
using ..Util.Devices: ones
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

export EnvOracle, check_oracle, UniformTicTacToeEnvOracle
export Policy, Tree, explore

# # Environment oracles

"""
    EnvOracle(; init_fn, transition_fn)

An environment oracle is defined by two functions: `init_fn` and `transition_fn`. These
functions operate on batches of states directly, enabling efficient parallelization on GPU,
CPU or both.

# The `init_fn` function

`init_fn` takes a vector of environment objects as an argument. Environment objects are of
the same type than those passed to the `explore` and `gumbel_explore` functions. The
`init_fn` function returns a named-tuple of same-size arrays with the following fields:

- `internal_states`: internal representations of the environment states as used by MCTS and
    manipulated by `transition_fn`. `internal_states` must be a single or multi-dimensional
    array whose last dimension is a batch dimension (see examples below).
- `valid_actions`: a vector of booleans with dimensions `num_actions` and `batch_id`
  indicating which actions are valid to take (this is disregarded in MuZero).
- `policy_prior`: the policy prior for each states as an `AbstractArray{Float32,2}` with
    dimensions `num_actions` and `batch_id`.
- `value_prior`: the value prior for each state as an `AbstractVector{Float32}`.

# The `transition_fn` function

`transition_fn` takes as arguments a vector of internal states (as returned by `init_fn` for
example) along with a vector of action ids. Action ids consist in integers between 1 and
`num_actions` and are valid indices for `policy_priors` and `value_priors`. The
`transition_fn` function returns a named-tuple of arrays:

- `internal_states`: new states reached after executing the proposed actions (see
    `init_fn`).
- `rewards`: vector of `Float32` indicating the intermediate rewards collected during the
    transitions.
- `terminal`: vector of booleans indicating whether or not the reached states are terminal
    (this is always `false` in MuZero).
- `valid_actions`: a vector of booleans with dimensions `num_actions` and `batch_id`
  indicating which actions are valid to take (this is disregarded in MuZero).
- `player_switched`: vector of booleans indicating whether or not the current player
    switched during the transition (always `true` in many board games).
- `policy_prior`, `value_prior`: same as for `init_fn`.


# Examples of internal state encodings

- When using AlphaZero on board games such as tictactoe, connect-four, Chess or Go, internal
  states are exact state encodings (since AlphaZero can access an exact simulator). The
  `internal_states` field can be made to have type `AbstractArray{Float32, 4}` with
  dimensions `player_channel`, `board_width`, `board_height` and `batch_id`. Alternatively,
  one can use a one-dimensional vector with element type `State` where
  `Base.isbitstype(State)`. The latter representation may be easier to work with when
  broadcasting non-batched environment implementations on GPU (see
  `Tests.Common.BitwiseTicTacToe.BitwiseTicTacToe` for example).
- When using MuZero, the `internal_states` field typically has type `AbstractArray{Float32,
  2}` where the first dimension corresponds to the size of latent states and the second
  dimension is the batch dimension.
"""
@kwdef struct EnvOracle{I<:Function,T<:Function}
    init_fn::I
    transition_fn::T
end

function check_keys(keys, ref_keys)
    return Set(keys) == Set(ref_keys)
end

"""
    check_oracle(::EnvOracle, env)

This function performs some sanity checks to see if an environment oracle is correctly
specified on a given environment instance.

A list of environments `envs` must be specified, along with a list of actions `aids`.

The function returns `nothing` if no problems are detected. Otherwise, helpful error
messages are raised.
"""
function check_oracle(oracle::EnvOracle, envs, aids)
    B = length(envs)

    @assert B == length(aids) "`envs` and `aids` must be of the same size."

    init_res = oracle.init_fn(envs)
    # Named-tuple check
    @assert check_keys(
        keys(init_res), (:internal_states, :valid_actions, :policy_prior, :value_prior)
    ) "The `EnvOracle`'s `init_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, valid_actions, policy_prior, " *
        "value_prior."

    # Type and dimensions check
    size_valid_actions = size(init_res.valid_actions)
    @assert (
        length(size_valid_actions) == 2 &&
        size_valid_actions[2] == B &&
        typeof(init_res.valid_actions[1]) == Bool
    ) "The `init_fn`'s function should return a `valid_actions` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Bool`."
    size_policy_prior = size(init_res.policy_prior)
    @assert (
        length(size_policy_prior) == 2 &&
        size_policy_prior[2] == B &&
        typeof(init_res.policy_prior[1]) == Float32
    ) "The `init_fn`'s function should return a `policy_prior` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Float32`."
    @assert (
        length(init_res.value_prior) == B && typeof(init_res.value_prior[1]) == Float32
    ) "The `init_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    transition_res = oracle.transition_fn(envs, aids)
    # Named-tuple check
    @assert check_keys(
        keys(transition_res),
        (
            :internal_states,
            :rewards,
            :terminal,
            :valid_actions,
            :player_switched,
            :policy_prior,
            :value_prior,
        ),
    ) "The `EnvOracle`'s `transition_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, rewards, terminal, valid_actions, " *
        "player_switched, policy_prior, value_prior."

    # Type and dimensions check
    @assert (
        length(transition_res.rewards) == B && typeof(transition_res.rewards[1]) == Float32
    ) "The `transition_fn`'s function should return a `rewards` vector of length " *
        "`batch_id` and of type `Float32`."
    @assert (
        length(transition_res.terminal) == B && typeof(transition_res.terminal[1]) == Bool
    ) "The `transition_fn`'s function should return a `terminal` vector of length " *
        "`batch_id` and of type `Bool`."
    size_valid_actions = size(transition_res.valid_actions)
    @assert (
        length(size_valid_actions) == 2 &&
        size_valid_actions[2] == B &&
        typeof(transition_res.valid_actions[1]) == Bool
    ) "The `transition_fn`'s function should return a `valid_actions` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Bool`."
    @assert (
        length(transition_res.player_switched) == B &&
        typeof(transition_res.player_switched[1]) == Bool
    ) "The `transition_fn`'s function should return a `player_switched` vector of length " *
        "`batch_id`, and of type `Bool`."
    size_policy_prior = size(transition_res.policy_prior)
    @assert (
        length(size_policy_prior) == 2 &&
        size_policy_prior[2] == B &&
        typeof(transition_res.policy_prior[1]) == Float32
    ) "The `transition_fn`'s function should return a `policy_prior` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Float32`."
    @assert (
        length(transition_res.value_prior) == B &&
        typeof(transition_res.value_prior[1]) == Float32
    ) "The `transition_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    return nothing
end

# ## Example Environment Oracle
# ### RandomWalk1D Environment Oracle
"""
    UniformTicTacToeEnvOracle()

Define an `EnvOracle` object with a uniform policy for the game of Tic-Tac-Toe.

This environment is a wrapper around the Tic-Tac-Toe environment of RL.jl.
For more details, checkout their documentation:
https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TicTacToeEnv
"""
function UniformTicTacToeEnvOracle()
    get_policy_prior(A, B) = ones(Float32, (A, B)) / A
    get_value_prior(B) = zeros(Float32, B)

    function init_fn(envs)
        A = length(action_space(envs[1]))
        B = length(envs)

        @assert B > 0
        @assert all(e -> length(action_space(e)) == A, envs)

        valid_actions = zeros(Bool, (A, B))
        for (bid, env) in enumerate(envs)
            valid_actions[:, bid] = legal_action_space_mask(env)
        end

        return (;
            internal_states=envs,
            valid_actions,
            policy_prior=get_policy_prior(A, B),
            value_prior=get_value_prior(B),
        )
    end

    function transition_fn(envs, aids)
        A = length(action_space(envs[1]))
        B = length(envs)

        prev_players = map(current_player, envs)
        internal_states = map(zip(envs, aids)) do (env, aid)
            next_env = copy(env)
            @assert aid in legal_action_space(env) "Tried to play an illegal move"
            next_env(aid)
            next_env
        end

        valid_actions = zeros(Bool, (A, B))
        for (bid, env) in enumerate(internal_states)
            valid_actions[:, bid] = legal_action_space_mask(env)
        end

        return (;
            internal_states,
            rewards=Float32.(reward.(internal_states, prev_players)),
            terminal=is_terminated.(internal_states),
            valid_actions,
            player_switched=prev_players .!= map(current_player, internal_states),
            policy_prior=get_policy_prior(A, B),
            value_prior=get_value_prior(B),
        )
    end
    return EnvOracle(; init_fn, transition_fn)
end

# # Policy definition

"""
    Policy(; kwds...)

# Keyword Arguments

TODO: document the keyword arguments.
"""
@kwdef struct Policy{Device,Oracle<:EnvOracle}
    device::Device
    oracle::Oracle
    num_simulations::Int = 64
    num_considered_actions::Int = 8
    value_scale::Float32 = 0.1f0
    max_visit_init::Int = 50
end

# # Tree datastructure

## Value stored in tree.parent for nodes with no parents
const NO_PARENT = Int16(0)
## Value stored in tree.children for unvisited children
const UNVISITED = Int16(0)

"""
A batch of MCTS trees, represented as a structure of arrays.

# Fields

We provide shape information between parentheses: `B` denotes the batch size, `N` the
maximum number of nodes (i.e. number of simulations) and `A` the number of actions.

## Tree structure and statistics

- `parent`: the id of the parent node or `NO_PARENT` (N, B)
- `num_visits`: number of times the node was visited (N, B)
- `total_values`: the sum of all values backpropagated to a node (N, B)
- `children`: node id of all children or UNVISITED for unvisited actions (A, N, B)

## Cached static information

All these fields are used to store the results of calling the environment oracle.

- `state`: state vector or embedding (..., N, B)
- `terminal`: whether a node is a terminal node (N, B)
- `valid_actions`: whether or not each action is valid or not (A, N, B)
- `prev_action`: the id of the action leading to this node or 0 (N, B)
- `prev_reward`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B)
- `prev_switched`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B)
- `policy_prior`: as given by the oracle (A, N, B)
- `value_prior`: as given by the oracle (N, B)

# Remarks

- The `Tree` structure is parametric in its field array types since those could be
  instantiated on CPU or GPU.
- It is yet to be determined whether a batch of MCTS trees is more cache-friendly when
  represented as a structure of arrays (as is the case here) or as an array of structures
  (as in the `BatchedMctsAos` implementation).
- It is yet to be determined whether or not permuting the `N` and `B` dimensions of all
  arrays would be more cache efficient. An `(N, B)` representation is more conventional, it
  is used in Deepmind's MCTX library (and might be imposed by JAX) and it may provide better
  temporal locality since each thread is looking at a different batch. On the other hand, a
  `(B, N)` layout may provide better spatial locality when copying the results of the
  environment oracle and possibly when navigating trees.

# TODO: add a comment on CuArray vs Arrayfor the parameter of Tree
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
    parent::Int16NodeArray
    num_visits::Int16NodeArray
    total_values::Float32NodeArray
    children::Int16ActionArray
    ## Cached oracle info
    state::StateNodeArray
    terminal::BoolNodeArray
    valid_actions::BoolActionArray
    prev_action::Int16NodeArray
    prev_reward::Float32NodeArray
    prev_switched::BoolNodeArray
    policy_prior::Float32ActionArray
    value_prior::Float32NodeArray
end

## https://cuda.juliagpu.org/stable/tutorials/custom_structs/
@adapt_structure Tree

function validate_prior(policy_prior, valid_actions)
    prior = map(zip(policy_prior, valid_actions)) do (prior, is_valid)
        (is_valid) ? prior : Float32(0)
    end
    prior_sum = mapslices(prior; dims=1) do prior_slice
        sum(prior_slice; init=0)
    end
    @assert !all(prior_sum .== 0) "No available actions"
    return @. prior / prior_sum
end

function create_tree(mcts, envs)
    info = mcts.oracle.init_fn(envs)
    A, N, B = size(info.policy_prior)[1], mcts.num_simulations, length(envs)

    num_visits = zeros(Int16, mcts.device, (N, B))
    num_visits[1, :] .= 1
    internal_states = DeviceArray(mcts.device){AbstractEnv}(undef, (N, B))
    internal_states[1, :] = info.internal_states
    valid_actions = zeros(Bool, mcts.device, (A, N, B))
    valid_actions[:, 1, :] = info.valid_actions
    policy_prior = zeros(Float32, mcts.device, (A, N, B))
    policy_prior[:, 1, :] = validate_prior(info.policy_prior, info.valid_actions)
    value_prior = zeros(Float32, mcts.device, (N, B))
    value_prior[1, :] = info.value_prior

    return Tree(;
        parent=zeros(Int16, mcts.device, (N, B)),
        num_visits,
        total_values=zeros(Float32, mcts.device, (N, B)),
        children=zeros(Int16, mcts.device, (A, N, B)),
        state=internal_states,
        terminal=zeros(Bool, mcts.device, (N, B)),
        valid_actions,
        prev_action=zeros(Int16, mcts.device, (N, B)),
        prev_reward=zeros(Float32, mcts.device, (N, B)),
        prev_switched=zeros(Bool, mcts.device, (N, B)),
        policy_prior,
        value_prior,
    )
end

function Base.size(tree::Tree)
    A, N, B = size(tree.children)
    return (; A, N, B)
end

batch_size(tree) = size(tree).B

# # MCTS implementation

value(tree, cid, bid) = tree.total_values[cid, bid] / tree.num_visits[cid, bid]

qvalue(tree, cid, bid) = value(tree, cid, bid) * (-1)^tree.prev_switched[cid, bid]

function get_visited_children(tree, cid, bid)
    direct_children = tree.children[:, cid, bid]
    direct_children = direct_children[direct_children .!= 0]
    return direct_children
end

function root_value_estimate(tree, cid, bid) # TODO
    total_qvalues = 0.0f0
    total_prior = 0.0f0
    total_visits = 0
    for (aid, cnid) in enumerate(get_visited_children(tree, cid, bid))
        total_qvalues += tree.policy_prior[aid, cid, bid] * qvalue(tree, cnid, bid)
        total_prior += tree.policy_prior[aid, cid, bid]
        total_visits += tree.num_visits[cnid, bid]
    end
    children_value = total_qvalues
    total_prior > 0 && (children_value /= total_prior)
    return (tree.value_prior[cid, bid] + total_visits * children_value) / (1 + total_visits)
end

function completed_qvalues(tree, cid, bid)
    root_value = root_value_estimate(tree, cid, bid)
    (; A) = size(tree)
    return map(1:A) do aid
        if (!tree.valid_actions[aid, cid, bid])
            return -Inf32
        end

        cnid = tree.children[aid, cid, bid]
        return cnid > 0 ? qvalue(tree, cnid, bid) : root_value
    end
end

function qcoeff(mcts, tree, cid, bid)
    # XXX: init is necessary for GPUCompiler right now...
    max_child_visit = maximum(get_visited_children(tree, cid, bid); init=Int16(0))
    return mcts.value_scale * (mcts.max_visit_init + max_child_visit)
end

function target_policy(mcts, tree, cid, bid)
    qs = completed_qvalues(tree, cid, bid)
    return softmax(log.(tree.policy_prior[:, cid, bid]) + qcoeff(mcts, tree, cid, bid) * qs)
end

function select_nonroot_action(mcts, tree, cid, bid)
    policy = target_policy(mcts, tree, cid, bid)
    num_child_visits = [
        (cnid != 0) ? tree.num_visits[cnid, bid] : 0 for cnid in tree.children[:, cid, bid]
    ]
    total_visits = sum(num_child_visits; init=0)
    return argmax(
        policy - Float32.(num_child_visits) / (total_visits + 1); init=(0, -Inf32)
    )
end

function select(mcts, tree, bid)
    cur = Int16(1)
    while true
        if tree.terminal[cur, bid]
            return cur
        end
        aid = select_nonroot_action(mcts, tree, cur, bid)
        cnid = tree.children[aid, cur, bid]
        if cnid > 0
            cur = cnid
        else
            return cur # returns parent
        end
    end
    return nothing
end

function select_and_eval!(mcts, tree, simnum)
    B = batch_size(tree)
    batch_indices = DeviceArray(mcts.device)(1:B)
    pfrontier = map(batch_indices) do bid
        select(mcts, tree, bid)
    end

    # Compute transition at `pfrontier`
    non_terminal_mask = .![tree.terminal[cid, bid] for (cid, bid) in zip(pfrontier, 1:B)]
    # No new node to expand (a.k.a only terminal node on the frontier)
    (!any(non_terminal_mask)) && return pfrontier

    parent_ids = pfrontier[non_terminal_mask]
    non_terminal_bids = OneTo(B)[non_terminal_mask]
    action_ids = [
        select_nonroot_action(mcts, tree, cid, bid) for
        (cid, bid) in zip(parent_ids, non_terminal_bids)
    ]
    parent_states = [
        tree.state[pid, bid] for (pid, bid) in zip(parent_ids, non_terminal_bids)
    ]
    info = mcts.oracle.transition_fn(parent_states, action_ids)

    # Create nodes and save `info`
    tree.parent[simnum, non_terminal_mask] = parent_ids
    tree.children[action_ids, parent_ids, non_terminal_mask] .= simnum
    tree.state[simnum, non_terminal_mask] = info.internal_states
    tree.terminal[simnum, non_terminal_mask] = info.terminal
    tree.valid_actions[:, simnum, non_terminal_mask] = info.valid_actions
    tree.prev_action[simnum, non_terminal_mask] = action_ids
    tree.prev_reward[simnum, non_terminal_mask] = info.rewards
    tree.prev_switched[simnum, non_terminal_mask] = info.player_switched
    tree.policy_prior[:, simnum, non_terminal_mask] = info.policy_prior
    tree.value_prior[simnum, non_terminal_mask] = info.value_prior

    # Update frontier
    frontier = pfrontier
    frontier[non_terminal_mask] .= simnum

    return frontier
end

function backpropagate!(mcts, tree, frontier)
    B = batch_size(tree)
    batch_ids = DeviceArray(mcts.device)(1:B)
    map(batch_ids) do bid
        sid = frontier[bid]
        val = tree.value_prior[sid, bid]
        while true
            tree.num_visits[sid, bid] += Int16(1)
            tree.total_values[sid, bid] += val
            if tree.parent[sid, bid] > 0
                (tree.prev_switched[sid, bid]) && (val = -val)
                val += tree.prev_reward[sid, bid]
                sid = tree.parent[sid, bid]
            else
                return nothing
            end
        end
    end
    return nothing
end

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; B, N) = size(tree)
    frontier = DeviceArray(mcts.device)(ones(Int16, B))
    for i in 2:N
        frontier = select_and_eval!(mcts, tree, i)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

end