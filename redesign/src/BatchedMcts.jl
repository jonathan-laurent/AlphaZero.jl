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
using Base: @kwdef
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

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
- `policy_priors`: the policy prior for each states as an `AbstractArray{Float32,2}` with
    dimensions `num_actions` and `batch_id`.
- `value_priors`: the value prior for each state as an `AbstractVector{Float32}`.

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
- `policy_priors`, `value_priors`: same as for `init_fn`.


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

"""
    check_oracle(::EnvOracle, env)

This function performs some sanity checks to see if an environment oracle is correctly
specified on a given environment instance.

The function returns `nothing` if no problems are detected. Otherwise, helpful error
messages are raised.
"""
function check_oracle(oracle::EnvOracle, env)
    # TODO
    return nothing
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

batch_size(tree) = size(tree)[1]

# # MCTS implementation

function select!(mcts, tree, simnum)
    B = batch_size(tree)
    batch_indices = DeviceArray(mcts.device)(1:B)
    frontier = map(batch_indices) do bid
        @assert false, "Not implemented"
    end
    return frontier
end

end
