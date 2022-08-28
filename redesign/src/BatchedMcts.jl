"""
    BatchedMcts

A batched implementation of MCTS that can run on CPU or GPU.


Check out the part "Core MCTS algorithm" if you want to know more about the MCTS algorithm.

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

The examples below assume that you run the following code before:
```jldoctest
julia> using RLZero
julia> using .Tests
```

First, we need to create a list of environments from which we would like to find the optimal
action. Let's choose the Tic-Tac-Toe game for our experiment.
```jldoctest
julia> envs = [bitwise_tictactoe_draw(), bitwise_tictactoe_winning()]
```

Here, it's worth noting that we used the bitwise versions of our position-specific
environments in `./Tests/Common/BitwiseTicTacToe.jl`. Those environments are here to ease
the experimentations and the tests of the package. Bitwise versions are sometimes necessary
to comply with GPU constraints. But in this case, the only motivation to choose them was
the compatibility it offers with `UniformTicTacToeEnvOracle`.

In fact, any environment can be used in `BatchedMcts` if we provide the appropriate
environment oracle. See `EnvOracle` for more details on this.

We should then provide a `Policy` to the Mcts. There are most noticeably two arguments to
provide: `device` and `oracle`. The `device` specifies where the algorithm should run. Do
you want it to run on the `CPU` or the `GPU`? It's straightforward. The `oracle`
arguments is an `EnvOracle`. You can use the default provided one,
`UniformTicTacToeEnvOracle` or create your one for other games. For the latter, do not
hesitate to check `EnvOracle` and `check_oracle`.

The `Policy` also accepts other arguments. Refers to the corresponding section to know more.
```jldoctest
julia> policy = BatchedMcts.Policy(;
    device=GPU(),
    oracle=BatchedMcts.UniformTicTacToeEnvOracle()
)
```

After those 2 simple steps, we can now call the `explore` function to find out the optimal
action to choose. This implementation provides two MCTS exploration implementations:
`explore` & `gumbel_explore`. In the context of AlphaZero/ MuZero, each of them is more
adapted to a specific context:
- `gumbel_explore` is more suited for the training context of AlphaZero/ MuZero. It encourages
   exploring slightly sub-optimal actions and thus offers more diversity of
   game positions to the neural network.
- `explore`, on the other hand, is more suited for the inference context. No noise is added
   to the exploration. It therefore hopefully finds the optimal policy.

Therefore, if you are only interested in the optimal action, always use the `explore`
function.
```jldoctest
julia> tree = BatchedMcts.explore(policy, envs)
```

If you are interested in the exploration undergone, you can check the `Tree` structure.
Otherwise, a simple call to `completed_qvalues` will give you a comprehensive score of how
good each action is. The higher the better of course. We can then use the `argmax` utility
to pick up the best action.
```jldoctest
julia> qs = BatchedMcts.completed_qvalues(tree)
julia> argmax.(qs) # The optimal action for each environment
```

This implementation of batched Mcts tries to provide flexible interfaces to run code on any
device. You can easily run the tree search on `CPU` or `GPU` with the `device` argument
of `Policy`. If you want the state evaluation or the environment simulation to run on
GPU as well, you can! This will be handled in the `EnvOracle` definition.

By default, `UniformTicTacToeEnvOracle`'s `transition_fn` runs on both `CPU` and `GPU`
depending on the array type of `envs` (a.k.a GPU's `CuArray` vs classic CPU's `Array`). To
write your custom state evaluation or environment simulation on the appropriate device,
check `EnvOracle` and its example `UniformTicTacToeEnvOracle`.

TODO: This section should show examples of using the module (using jltest?). Ideally, it
should demonstrate different settings such as:

- An AlphaZero-like setting where everything runs on GPU.
- An AlphaZero-like setting where state evaluation runs on GPU and all the rest runs on CPU
  (tree search and environment simulation).
- A MuZero-like setting where everything runs on GPU.

This section should also demonstrate the use of the `check_policy` function to perform
sanity checks on user environments.


# Naming conventions

Here is a short list of variable names we used throughout this file and what they mean, if
it is not obvious:
- bid: buffer index, used to index the batch (`B`) dimension.
- cid: current simulation index, used to index the simulation (`N`) dimension.
- cnid: child index, used to index the simulation (`N`) dimension as well.
- aid: action index, used to index the action (`A`) dimension.
- aids: action index list, used to index the action (`A`) dimension.
- qs: completed qvalues.
- simnum: simulation number (which is also a simulation index).


# References

- Reference on the Gumbel MCTS extension:
  https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel
"""
module BatchedMcts

using Adapt: @adapt_structure
using Base: @kwdef, size
using Distributions: Gumbel
using Random: AbstractRNG
import Base.Iterators.map as imap
using StaticArrays
using CUDA: @inbounds
using EllipsisNotation

using ..BatchedEnvs
using ..Util.Devices
using ..Util.Devices.KernelFuns: sum, argmax, maximum, softmax

include("./Tests/Common/BitwiseTicTacToe.jl")
using .BitwiseTicTacToe

export EnvOracle, check_oracle, UniformTicTacToeEnvOracle
export Policy, Tree, explore, gumbel_explore, completed_qvalues

# # Environment oracles

"""
    EnvOracle(; init_fn, transition_fn)

An environment oracle is defined by two functions: `init_fn` and `transition_fn`. These
functions operate on batches of states directly, enabling efficient parallelization on GPU,
CPU or both.

# The `init_fn` function

`init_fn` takes a vector of environment objects as an argument. Environment objects are of
the same type as those passed to the `explore` and `gumbel_explore` functions. The
`init_fn` function returns a named-tuple of same-size arrays with the following fields:

- `internal_states`: internal representations of the environment states as used by MCTS and
   manipulated by `transition_fn`. `internal_states` must be a single or multi-dimensional
   array whose last dimension is a batch dimension (see examples below). `internal_states`
   must be `isbits` if it is desired to run `BatchedMcts` on `GPU`.
- `valid_actions`: a vector of booleans with dimensions `num_actions` and `batch_id`
   indicating which actions are valid to take (this is disregarded in MuZero).
- `policy_prior`: the policy prior for each state as an `AbstractArray{Float32,2}` with
   dimensions `num_actions` and `batch_id` and value within [0, 1]. An action with
   `policy_prior` of `0` corresponds to the worst possible one or an illegal action. On the
   other side, a value of `1` is associated with a winning action.
- `value_prior`: the value prior for each state as an `AbstractVector{Float32}` and value
   within [-1, 1]. In the same way than `policy_prior`, -1 and 1 respectively correspond to 
   a bad and a good position.

# The `transition_fn` function

`transition_fn` takes as arguments a vector of internal states (as returned by `init_fn`)
along with a vector of action ids. Action ids consist in integers between 1 and
`num_actions` and are valid indices for `policy_prior` and `value_prior`. 
    
Note that `init_fn` will always receive the same array as the one passed to `explore` or
`gumbel_explore` as `envs` (which should be a CPU `Array`). But it's a bit more tricky for
`transition_fn`. It may receive both CPU `Array` or GPU `CuArray` depending on the device
specified in `Policy`. To handle both more easily look at `Util.Devices` and how it 
is used in `UniformTicTacToeEnvOracle`.

In the context of a `Policy` on the GPU, `transition_fn` can both return a CPU `Array` or a
GPU `CuArray`. The `CuArray` is more adapted as it will prevent memory transfers, but both
works.

The `transition_fn` function returns a named-tuple of arrays:

- `internal_states`: new states reached after executing the proposed actions (see
   `init_fn`).
- `rewards`: vector of `Float32` indicating the intermediate rewards collected during the
   transitions.
- `terminal`: vector of booleans indicating whether or not the reached states are terminal
   (this is always `false` in MuZero).
- `player_switched`: vector of booleans indicating whether or not the current player
   switched during the transition (always `true` in many board games).
- `valid_actions`, `policy_prior`, `value_prior`: same as for `init_fn`.


# Examples of internal state encodings

- When using AlphaZero on board games such as tictactoe, connect-four, Chess or Go, internal
  states are exact state encodings (since AlphaZero can access an exact simulator). The
  `internal_states` field can be made to have type `AbstractArray{Float32, 4}` with
  dimensions `player_channel`, `board_width`, `board_height` and `batch_id`. Alternatively,
  one can use a one-dimensional vector with element type `State` where
  `Base.isbitstype(State)`. The latter representation may be easier to work with when
  broadcasting non-batched environment implementations on GPU (see
  `Tests.Common.BitwiseTicTacToe.BitwiseTicTacToe` for example).
- When using MuZero, the `internal_states` field typically has the type `AbstractArray{Float32,
  2}` where the first dimension corresponds to the size of latent states and the second
  dimension is the batch dimension.

See also [`check_oracle`](@ref), [`UniformTicTacToeEnvOracle`](@ref)
"""
@kwdef struct EnvOracle{I<:Function,T<:Function}
    init_fn::I
    transition_fn::T
end

"""
    check_keys(keys, ref_keys)

Check that the two lists of symbols, `keys` and `ref_keys`, are identical.

Small utilities used  in `check_oracle` to compare keys of named-tuple.
"""
function check_keys(keys, ref_keys)
    return Set(keys) == Set(ref_keys)
end

"""
    check_oracle(::EnvOracle, env)

Perform some sanity checks to see if an environment oracle is correctly specified on a
given environment instance.

A list of environments `envs` must be specified, along with the `EnvOracle` to check.

Return `nothing` if no problems are detected. Otherwise, helpful error messages are raised.
More precisely, `check_oracle` verifies the keys of the returned named-tuples from `init_fn`
a `transition_fn` and the types and dimensions of their lists.

See also [`EnvOracle`](@ref)
"""
function check_oracle(oracle::EnvOracle, envs)
    B = length(envs)

    init_res = oracle.init_fn(envs)
    # Named-tuple check
    @assert check_keys(
        keys(init_res), (:internal_states, :valid_actions, :policy_prior, :value_prior)
    ) "The `EnvOracle`'s `init_fn` function should returned a named-tuple with the " *
        "following fields: internal_states, valid_actions, policy_prior, " *
        "value_prior."

    # Type and dimensions check
    size_valid_actions = size(init_res.valid_actions)
    A, _ = size_valid_actions
    @assert (size_valid_actions == (A, B) && eltype(init_res.valid_actions) == Bool) "The " *
        "`init_fn`'s function should return a `valid_actions` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Bool`."
    size_policy_prior = size(init_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(init_res.policy_prior) == Float32) "The " *
        "`init_fn`'s function should return a `policy_prior` vector with dimensions " *
        "`num_actions` and `batch_id`, and of type `Float32`."
    @assert (length(init_res.value_prior) == B && eltype(init_res.value_prior) == Float32) "The " *
        "`init_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    aids = [
        findfirst(init_res.valid_actions[:, bid]) for
        bid in 1:B if any(init_res.valid_actions[:, bid])
    ]
    envs = [env for (bid, env) in enumerate(envs) if any(init_res.valid_actions[:, bid])]

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
        length(transition_res.rewards) == B && eltype(transition_res.rewards) == Float32
    ) "The `transition_fn`'s function should return a `rewards` vector of length " *
        "`batch_id` and of type `Float32`."
    @assert (
        length(transition_res.terminal) == B && eltype(transition_res.terminal) == Bool
    ) "The `transition_fn`'s function should return a `terminal` vector of length " *
        "`batch_id` and of type `Bool`."
    size_valid_actions = size(transition_res.valid_actions)
    @assert (size_valid_actions == (A, B) && eltype(transition_res.valid_actions) == Bool) "The `" *
        "transition_fn`'s function should return a `valid_actions` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Bool`."
    @assert (
        length(transition_res.player_switched) == B &&
        eltype(transition_res.player_switched) == Bool
    ) "The `transition_fn`'s function should return a `player_switched` vector of length " *
        "`batch_id`, and of type `Bool`."
    size_policy_prior = size(transition_res.policy_prior)
    @assert (size_policy_prior == (A, B) && eltype(transition_res.policy_prior) == Float32) "The " *
        "`transition_fn`'s function should return a `policy_prior` vector with " *
        "dimensions `num_actions` and `batch_id`, and of type `Float32`."
    @assert (
        length(transition_res.value_prior) == B &&
        eltype(transition_res.value_prior) == Float32
    ) "The `transition_fn`'s function should return a `value_policy` vector of length " *
        "`batch_id`, and of type `Float32`."

    return nothing
end

# ## Example Environment Oracle
# ### Tic-Tac-Toe Environment Oracle with a Uniform policy

"""
    UniformTicTacToeEnvOracle()

Define an `EnvOracle` object with a uniform policy for the game of Tic-Tac-Toe.

This oracle environment is a wrapper around the `Tests.Common.BitwiseTicTacToeEnv`.
It can be both used on `CPU` & `GPU`.

It was inspired by the RL.jl library. For more details, check out their documentation:
https://juliareinforcementlearning.org/docs/rlenvs/#ReinforcementLearningEnvironments.TicTacToeEnv

See also [`EnvOracle`](@ref)
"""
function UniformTicTacToeEnvOracle()
    A = 9 # Number of case in the Tic-Tac-Toe grid

    get_policy_prior(A, B, device) = ones(Float32, device, (A, B)) / A
    get_value_prior(B, device) = zeros(Float32, device, B)

    function get_valid_actions(A, B, envs)
        device = get_device(envs)

        valid_ids = DeviceArray(device)(Tuple.(CartesianIndices((A, B))))
        return map(valid_ids) do (aid, bid)
            valid_action(envs[bid], aid)
        end
    end

    function init_fn(envs)
        B = length(envs)
        device = get_device(envs)

        @assert B > 0
        @assert all(@. num_actions(envs) == A)

        return (;
            internal_states=envs,
            valid_actions=get_valid_actions(A, B, envs),
            policy_prior=get_policy_prior(A, B, device),
            value_prior=get_value_prior(B, device),
        )
    end

    # Utilities to access elements returned by `act` in `transition_fn`.
    # `act` return a tuple with the following form:
    #   (state, (; reward, switched))
    get_state(info) = first(info)
    get_reward(info) = last(info).reward
    get_switched(info) = last(info).switched

    function transition_fn(envs, aids)
        B = length(envs)
        device = get_device(envs)

        @assert all(valid_action.(envs, aids)) "Tried to play an illegal move"

        act_info = act.(envs, aids)
        player_switched = get_switched.(act_info)
        rewards = Float32.(get_reward.(act_info))
        internal_states = get_state.(act_info)

        return (;
            internal_states,
            rewards,
            terminal=terminated.(internal_states),
            valid_actions=get_valid_actions(A, B, internal_states),
            player_switched,
            policy_prior=get_policy_prior(A, B, device),
            value_prior=get_value_prior(B, device),
        )
    end
    return EnvOracle(; init_fn, transition_fn)
end

# # Policy definition

"""
    Policy{Device, Oracle<:EnvOracle}(; 
        device::Device
        oracle::Oracle
        num_simulations::Int = 64
        num_considered_actions::Int = 8
        value_scale::Float32 = 0.1f0
        max_visit_init::Int = 50
    )

A batch, device-specific MCTS Policy that leverages an external `EnvOracle`.


# Keyword Arguments

- `device::Device`: device on which the policy should preferably run (i.e. `CPU` or `GPU`).
- `oracle::Oracle`: environment oracle handling the environment simulation and the state
   evaluation.
- `num_simulations::Int = 64`: number of simulations to run on the given Mcts `Tree`.
- `num_considered_actions::Int = 8`: number of actions considered by Gumbel during
   exploration. Only the `num_conidered_actions` actions with the highest scores will be
   explored. It should preferably be a power of 2.
- `value_scale::Float32 = 0.1f0`: multiplying coefficient to weight the qvalues against the
   prior probabilities during exploration. Prior probabilities have, by default, a
   decreasing weight when the number of visits increases.
- `max_visit_init::Int = 50`: artificial increase of the number of visits to weight qvalue
   against prior probabilities on low visit count.

# Notes

The attributes `num_conidered_actions`, `value_scale` and `max_visit_init` are specific to
the Gumbel implementation.
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

## Value stored in `tree.parent` for nodes with no parents.
const NO_PARENT = Int16(0)
## Value stored in `tree.children` for unvisited children.
const UNVISITED = Int16(0)
## Value used in various tree's attributes to access root information.
const ROOT = Int16(1)
## Valud used when no action is selected.
const NO_ACTION = Int16(0)
# 2 Utilities to easily access information in `parent_frontier`
# `parent_frontier` is a list of tuples with the following format:
#   (parent, action)
const PARENT = Int16(1)
const ACTION = Int16(2)

"""
A batch of MCTS trees, represented as a structure of arrays.

# Fields

We provide shape information between parentheses: `B` denotes the batch size, `N` the
maximum number of nodes (i.e. number of simulations) and `A` the number of actions.

## Tree structure and statistics

- `parent`: the id of the parent node or `NO_PARENT` (N, B).
- `num_visits`: number of times the node was visited (N, B).
- `total_values`: the sum of all values backpropagated to a node (N, B).
- `children`: node id of all children or `UNVISITED` for unvisited actions (A, N, B).

## Cached static information

All these fields are used to store the results of calling the environment oracle.

- `state`: state vector or embedding as returned by `init_fn` of the `EnvOracle` (..., N, B).
- `terminal`: whether a node is a terminal node (N, B).
- `valid_actions`: whether or not each action is valid or not (A, N, B).
- `prev_action`: the id of the action leading to this node or 0 (N, B).
- `prev_reward`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B).
- `prev_switched`: the immediate reward obtained when transitioning from the parent from the
   perspective of the parent's player (N, B).
- `policy_prior`: as given by the `EnvOracle` (A, N, B).
- `value_prior`: as given by the `EnvOracle` (N, B).

# Remarks

- The `Tree` structure is parametric in its field array types since those could be
  instantiated on CPU or GPU (e.g. `Array{Bool, 3}` or 
  `CuArray{Bool, 1, CUDA.Mem.DeviceBuffer}` for `BoolActionArray`). See `create_tree` for
  more details on how a `Tree` is created.
- It is yet to be determined whether a batch of MCTS trees is more cache-friendly when
  represented as a structure of arrays (as is the case here) or as an array of structures
  (as in the `BatchedMctsAos` implementation).
- It is yet to be determined whether or not permuting the `N` and `B` dimensions of all
  arrays would be more cache efficient. An `(N, B)` representation is more conventional, it
  is used in Deepmind's MCTX library (and might be imposed by JAX) and it may provide better
  temporal locality since each thread is looking at a different batch. On the other hand, a
  `(B, N)` layout may provide better spatial locality when copying the results of the
  environment oracle and possibly when navigating trees.
- To complete the previous point, keep in mind that Julia is column-major (compared to the
  row-major, more classical paradigm, in most programming language like Python). This has the
  noticeable importance that the first dimension of a Matrix is continuous. The order of
  dimensions is then reversed compared to a Python implementation (like MCTX) to keep the
  same cache locality.
- It might seem weird at first to have a doubly linked tree (i.e. with both `children` and
  `parent` attributes), but is necessary to backpropagate values (`total_visits` and
  `total_values`).

See more about backpropagation in "Core MCTS algorithm".
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

l1_normalise(policy) = policy / abs(sum(policy; init=Float32(0)))

"""
    validate_prior(policy_prior, valid_actions)

Correct `policy_prior` to ignore in`valid_actions`.

More precisely, `policy_prior`  that are in`valid_actions` are set to 0. The rest of the
`policy_prior` (which are valid actions) are then l1-normalized.
"""
function validate_prior(policy_prior, valid_actions)
    valid_prior = policy_prior .* valid_actions
    # @assert begin
    #     B = last(size(valid_prior))
    #     all(any(valid_prior[:, bid] .!= Float32(0)) for bid in 1:B)
    # end "No available actions"
    return mapslices(l1_normalise, valid_prior; dims=1)
end

"""
    dims(arr::AbstractArray)
    dims(_)

Return the dimensions of an object.

This utility is used inside `create_tree` so that non-array objects have no dimension, rather
than popping an error as `size` do.
"""
dims(arr::AbstractArray) = size(arr)
dims(_) = ()

"""
    create_tree(mcts, envs)
    
Create a `Tree`.

Note that the `ROOT` of the `Tree` is considered explored, as a call to `init_fn` is done
on them. Moreover, `policy_prior` are corrected as specified in `validate_prior`.

See [`Tree`](@ref) for more details.
"""
function create_tree(mcts, envs)
    @assert length(envs) != 0 "There should be at least environment"

    info = mcts.oracle.init_fn(envs)
    A, N, B = size(info.policy_prior)[1], mcts.num_simulations, length(envs)

    num_visits = fill(UNVISITED, mcts.device, (N, B))
    num_visits[ROOT, :] .= 1
    internal_states = DeviceArray(mcts.device){eltype(info.internal_states)}(
        undef, (dims(info.internal_states[1])..., N, B)
    )
    internal_states[.., ROOT, :] = info.internal_states
    valid_actions = fill(false, mcts.device, (A, N, B))
    valid_actions[:, ROOT, :] = info.valid_actions
    policy_prior = zeros(Float32, mcts.device, (A, N, B))
    policy_prior[:, ROOT, :] = validate_prior(info.policy_prior, info.valid_actions)
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
        policy_prior,
        value_prior,
    )
end

"""
    Base.size(tree::Tree)

Return the number of actions (`A`), the number of simulations (`N`), and the number of
environments in the batch (`B`) of a `tree` as named-tuple `(; A, N, B)`.
"""
function Base.size(tree::Tree)
    A, N, B = size(tree.children)
    return (; A, N, B)
end

"""
    batch_size(tree)

Return the number of environments in the batch of a `tree`.
"""
batch_size(tree) = size(tree).B

# # MCTS implementation

# ## Basic MCTS functions

"""
    value(tree, cid, bid)

Return the absolute value of a game position.

The formula for a given node is:
    (prior_value + total_rewards) / num_visits

With `prior_value` the value as estimated by the oracle, `total_rewards` the sum of rewards
obtained from episodes including this node during exploration and `num_visits` the number of
episodes including this node.

See also [`qvalue`](@ref)
"""
value(tree, cid, bid) = tree.total_values[cid, bid] / tree.num_visits[cid, bid]

"""
    qvalue(tree, cid, bid)

Return the value of the game position from the perspective of its parent node.
I.e. a good position for your opponent is a bad one for you

See also [`value`](@ref)
"""
qvalue(tree, cid, bid) = value(tree, cid, bid) * (-1)^tree.prev_switched[cid, bid]

"""
    root_value_estimate(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}

Compute a value estimation of a node at this stage of the exploration.

The estimation is based on its `value_prior`, its number of visits, the `qvalue` of its
children and their associated `policy_prior`.
"""
function root_value_estimate(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}
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
    completed_qvalues(tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    
Return a list of estimated qvalue of all children for a given node.

More precisely, if its child have been visited at least one time, it computes its real
`qvalue`, otherwise, it uses the `root_value_estimate` of node instead.
"""
function completed_qvalues(
    tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}; invalid_actions_value=-Inf32
) where {A}
    root_value = root_value_estimate(tree, cid, bid, tree_size)
    ret = imap(1:A) do aid
        (!tree.valid_actions[aid, cid, bid]) && return invalid_actions_value

        cnid = tree.children[aid, cid, bid]
        return cnid != UNVISITED ? qvalue(tree, cnid, bid) : root_value
    end
    return SVector{A}(ret)
end

"""
    completed_qvalues(tree)
    
Return the `completed_qvalues` for each environments.
    
It is a practical utility functions to get a usable policy after a `explore`.
"""
function completed_qvalues(tree)
    ROOT = 1
    (; A, N, B) = BatchedMcts.size(tree)
    tree_size = (Val(A), Val(N), Val(B))

    return map(1:B) do bid
        q_values = BatchedMcts.completed_qvalues(
            tree, ROOT, bid, tree_size; invalid_actions_value=-1
        )
        l1_normalise(q_values)
    end
end

"""
    get_num_child_visits(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}

Return the number of visits of each child from the given node.
"""
function get_num_child_visits(tree, cid, bid, ::Tuple{Val{A},Any,Any}) where {A}
    ret = imap(1:A) do aid
        cnid = tree.children[aid, cid, bid]
        (cnid != UNVISITED) ? tree.num_visits[cnid, bid] : UNVISITED
    end
    return SVector{A}(ret)
end

"""
    qcoeff(mcts, tree, cid, bid, tree_size)

Compute a GUmbel-related ponderation of `qvalue`.

Through time, as the number of visits increases, the influence of `qvalue` builds up
relatively to `policy_prior`.
"""
function qcoeff(mcts, tree, cid, bid, tree_size)
    # XXX: init is necessary for GPUCompiler right now...
    max_child_visit = maximum(
        get_num_child_visits(tree, cid, bid, tree_size); init=UNVISITED
    )
    return mcts.value_scale * (mcts.max_visit_init + max_child_visit)
end

"""
    target_policy(mcts, tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    
Return the policy of a node.

I.e. a score of how much each action should be played. The higher, the better.
"""
function target_policy(mcts, tree, cid, bid, tree_size::Tuple{Val{A},Any,Any}) where {A}
    qs = completed_qvalues(tree, cid, bid, tree_size)
    policy = SVector{A}(imap(aid -> tree.policy_prior[aid, cid, bid], 1:A))
    return log.(policy) + qcoeff(mcts, tree, cid, bid, tree_size) * qs
end

"""
    select_action(mcts, tree, cid, bid, tree_size)

Select the most appropriate action to explore.

`NO_ACTION` is returned if no actions are available.
"""
function select_action(mcts, tree, cid, bid, tree_size)
    policy = softmax(target_policy(mcts, tree, cid, bid, tree_size))
    num_child_visits = get_num_child_visits(tree, cid, bid, tree_size)
    total_visits = sum(num_child_visits; init=UNVISITED)
    return Int16(
        argmax(
            policy - Float32.(num_child_visits) / (total_visits + 1);
            init=(NO_ACTION, -Inf32),
        ),
    )
end

# ## Core MCTS algorithm

"""
Let's now dive into the core part of the MCTS algorithm.

"MCTS" stands for "Monte Carlo Tree Search" and is a heuristic tree search algorithm, most
noticeably applied in the context of board games. For the record, recent breakthroughs in
the Reinforcement Learning fields applied MCTS to solve arcade games (e.g. Deepmind's MuZero)
or inside Tesla's autopilot software.

The tree search algorithms family focuses on finding the optimal policy (i.e. the best move to
play given a certain game position). Compared to other tree search algorithms, like the
simple MinMax algorithm, MCTS only explores a subset of the total search space. It does so
through exploration/ exploitation heuristics that orient this search toward the most promising
actions. Since its creation, MCTS has been shown to converge toward the MinMax algorithm as
the number of simulations increases, but at a much lower computational cost.

Originally, the MCTS algorithm was based on a Monte Carlo method. This method consisted of a
random sampling of the search space to evaluate the current board position. Those were
called "rollouts". Its creator, Bruce Abramson, attributes the rollouts "to be precise,
accurate, easily estimable, efficiently calculable, and domain-independent”. With the rise
of Deep Reinforcement Learning, rollouts were eventually replaced by Neural Networks. Those
were shown to be a way more precise evaluation method for complex board games. Though
modern MCTS does not involve any Monte Carlo methods anymore, it has still kept its name.
    

The MCTS iteratively runs simulations that expand a tree of explored game positions. Each
iteration can be divided into 3 phases:
- Selection: Start at the root node (initial board state) and walk to the child that has
  the best exploration/ exploitation tradeoff. This tradeoff was the source of a lot of
  research and has been formalized through different formulas, the most famous being UCB.
  The walk recursion is done until you hit a leaf node (i.e. a board game position that
  has never been explored or a terminal node like a win or a defeat). This reached leaf 
  node is then saved for the next phase.
- Evaluation: Sometimes split into two sub-phases, "Simulation" & "Expansion", the 
  evaluation phase respectively evaluates the current board position (either by a rollout in
  classic MCTS or through a Neural Network for a more modern one) and generates the children
  board position associated with each action.
- Backpropagation: After simulating a game by iteratively choosing the most promising
  actions (called `Episode`), and acquired some (intermediate or final) reward through its
  the last action, it is now needed to backpropagate this information up in the tree until
  the root node. The backpropagation updates the visit counts and total value of each node
  in this episode.
"""

function explore(mcts, envs)
    tree = create_tree(mcts, envs)
    (; N) = size(tree)
    for simnum in 2:N
        parent_frontier = select(mcts, tree)
        frontier = eval!(mcts, tree, simnum, parent_frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

"""
But in what way is this implementation exactly batched?

Julia has a great CUDA API. It is sufficiently high-level so that the same Julia code can be
both compiled to CPU and GPU according to the context. As an example, a function `map`ped
over a CUDA `CuArray` will be implicitly compiled and run on the GPU. In the same way, the
same function `map`ped over a classic CPU `Array`, will then be run on the CPU. It is as
simple as that. We extensively used this feature in AlphaZero.jl to extend code with GPU
support.

This principle brings both simplicity to the code (as no duplication of code is needed to
run on CPU and GPU) but also computing power to easily run code on GPU. This was one of
the reasons, Julia was chosen for AlphaZero.jl. We hope to make this implementation accessible
for students, researchers, and hackers while also being sufficiently powerful and fast to
enable meaningful experiments on limited computing resources.

The compilation on GPU still constraints a bit the way code is written. The following three
constraints should be respected:
- Data copied to GPU should be `isbits`.
- No dynamic allocation can be done on the GPU.
- Functions compiled on GPU should be type-stable.


To respect the `isbits` constraint, we used `SVector` from `StaticArrays` in GPU functions
to replace the more standard `Base.Vector`. To respect the type-stability constraints,
a `tree_size` is used as Value-as-parameter, so that size of `SVector` is known at
compile time. Some utility functions are provided in `Devices` module for easier use.


And as you can see in `select` below, the parallelization is done over the environments
list. In other words, a `select` parallel call is done over each environment.

In the same way, `backpropagate!` launch GPU kernels over the environments list. Those are
the only two functions to launch kernels
"""
function select(mcts, tree)
    (; A, N, B) = size(tree)
    tree_size = (Val(A), Val(N), Val(B))

    parent_frontier = zeros(Int16, mcts.device, (2, B))
    Devices.foreach(1:B, mcts.device) do bid
        new_frontier = select(mcts, tree, bid, tree_size)
        @inbounds parent_frontier[PARENT, bid] = new_frontier[PARENT]
        @inbounds parent_frontier[ACTION, bid] = new_frontier[ACTION]
        return nothing
    end
    return parent_frontier
end

"""
As a reminder, the selection phase walks through the tree to find a frontier of nodes to
expand. The selection starts at the root node (initial board state) and walks to the child
that has the best exploration/ exploitation tradeoff with the `select_action` utility function.
The walk recursion is done until you hit a leaf node (i.e. a board game position that has
never been explored or a terminal node like a win or a defeat). This reached leaf node is
then saved for the next phase.

There is a subtlety here compared to BatchedMCtsAos version. Nodes are not created at
the `selection`, because the concept of Oracle evaluation and environment simulation are
grouped in the `EnvOracle`. Those are split in BatchedMctsAos. Therefore, the `select` function
must return parent nodes of the frontier instead. That's why its returned value is caught
as `parent_frontier` in `explore`. The action chosen is also returned along with the parent
as a tuple, so that it prevents its recalculation.

But this leaves the question of the return value for terminal nodes... It has been chosen
to return the terminal nodes themself (instead of their parents) with a `NO_ACTION` as the
action in the tuple. This was necessary to respect the type-stability constraint. This way
`select` always returns a tuple `(node, action)`. The `NO_ACTION` action also makes it easy
to detect terminal nodes in the `parent_frontier`
"""
function select(mcts, tree, bid, tree_size; start=ROOT)
    cur = start
    while true
        if tree.terminal[cur, bid]
            # returns current terminal, but no action played
            return cur, NO_ACTION
        end
        aid = select_action(mcts, tree, cur, bid, tree_size)
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
The evaluation phase evaluates the current board position and simulates the environment with
the `EnvOracle` through the `transition_fn` interface and then saved information associated
to each action (i.e. `valid_actions` & `policy_prior`) along with the newly created nodes.

The `transition_fn` call takes a vector of state encodings and a vector of actions as
arguments. It therefore could not be parallelized over environments as `select` and
`backpropagate!` are.

Note that if all nodes at the frontier are terminal ones, then no call to `transition_fn` is
done (as no node needs to be created). The terminal nodes are then directly returned.

Lastly, to save `state` returned from `transition_fn`, it was needed to easily handle both
exact state encodings (like in AlphaZero) as well as latent space state encodings (like in
MuZero, encoding which will then have at least 1 dimension i.e. is a vector or a matrix).
To solve these constraints, we used the EllipsisNotation ("..") which was revealed to be
particularly useful. The EllipsisNotation enables to handle of any number of dimensions.
"""
function eval!(mcts, tree, simnum, parent_frontier)
    B = batch_size(tree)

    # Get terminal nodes at `parent_frontier`
    non_terminal_mask = parent_frontier[ACTION, :] .!= NO_ACTION
    non_terminal_bids = DeviceArray(mcts.device)(@view((1:B)[non_terminal_mask]))
    # No new node to expand (a.k.a only terminal node on the frontier)
    (length(non_terminal_bids) == 0) && return parent_frontier[PARENT, :]

    # Regroup `action_ids` and `parent_states` for `transition_fn`
    parent_ids = parent_frontier[PARENT, non_terminal_bids]
    action_ids = parent_frontier[ACTION, non_terminal_bids]

    state_cartesian_ids = CartesianIndex.(parent_ids, non_terminal_bids)
    parent_states = tree.state[.., state_cartesian_ids]
    info = mcts.oracle.transition_fn(parent_states, action_ids)

    # Create nodes and save `info`
    children_cartesian_ids = CartesianIndex.(action_ids, parent_ids, non_terminal_bids)

    @inbounds tree.parent[simnum, non_terminal_bids] = parent_ids
    @inbounds tree.children[children_cartesian_ids] .= simnum
    @inbounds tree.state[.., simnum, non_terminal_bids] = info.internal_states
    @inbounds tree.terminal[simnum, non_terminal_bids] = info.terminal
    @inbounds tree.valid_actions[:, simnum, non_terminal_bids] = info.valid_actions
    @inbounds tree.prev_action[simnum, non_terminal_bids] = action_ids
    @inbounds tree.prev_reward[simnum, non_terminal_bids] = info.rewards
    @inbounds tree.prev_switched[simnum, non_terminal_bids] = info.player_switched
    @inbounds tree.policy_prior[:, simnum, non_terminal_bids] = validate_prior(
        info.policy_prior, info.valid_actions
    )
    @inbounds tree.value_prior[simnum, non_terminal_bids] = info.value_prior

    # Update frontier
    frontier = parent_frontier[PARENT, :]
    @inbounds frontier[non_terminal_bids] .= simnum

    return frontier
end

"""
Finally, the backpropagation comes in. 

After simulating a game by iteratively choosing the most promising actions (which list of 
actions are called an `Episode`), and acquired some (intermediate or final) reward through
its last action, it is now needed to backpropagate this information up in the tree until
the root node. The backpropagation updates the visits count and total value of each node in
this episode.

The visits count of each node in the episode is simply incremented by 1. The total value of
each node is incremented by a certain value. This value is computed through TD learning
until the newly created node (i.e. the value of the newly created node estimated by the
oracle summed up the cumulative reward until this point). Of course, the sign of the TD
learned value must be switched when the player's turn is switched as well.

And as stated before, `backpropagate!` is parallelized in the same way as `select` (i.e.
over the environments list).
"""
function backpropagate!(mcts, tree, frontier)
    B = batch_size(tree)
    batch_ids = DeviceArray(mcts.device)(1:B)
    map(batch_ids) do bid
        cid = frontier[bid]
        val = tree.value_prior[cid, bid]
        while true
            val += tree.prev_reward[cid, bid]
            (tree.prev_switched[cid, bid]) && (val = -val)
            tree.num_visits[cid, bid] += Int16(1)
            tree.total_values[cid, bid] += val
            if tree.parent[cid, bid] != NO_PARENT
                cid = tree.parent[cid, bid]
            else
                return nothing
            end
        end
    end
    return nothing
end

# ### Gumbel MCTS variation

"""
This implementation provides two MCTS exploration implementations: `explore` &
`gumbel_explore`. In the context of AlphaZero/ MuZero, each of them is more adapted to a
specific context:
- `gumbel_explore` is more suited for the training context of AlphaZero/ MuZero. It
   encourages to explore of slightly sub-optimal actions and thus offers more diversity of
   game positions to the neural network.
- `explore`, on the other hand, is more suited for the inference context. No noise is added
   to the exploration. It therefore hopefully finds the optimal policy.

In the end, `explore` and `gumbel_explore` only differ from each other in a single line,
the use of `gumbel_select` instead of `select`.


The Gumbel algorithm won't be explained here in detail but you can learn more about it
here:
https://www.deepmind.com/publications/policy-improvement-by-planning-with-gumbel

The main idea behind `gumbel_explore` to successfully explore sub-optimal actions resides in
the addition of specific noise (the Gumbel noise) and a different simulations orchestration
in the `selection` phase.

Deepmind claims that this Gumbel variation does not pollute the policy later learned by the
Neural Network (on the contrary to Dirichlet noise used originally in AlphaZero/ MuZero).
This has the advantage to help the neural network learn faster with fewer simulations. It,
therefore, enables more meaningful experiments on limited computing resources.

This implementation is largely inspired by Deepmind's MCTX. It has the particularity to 
precompute the simulation orchestration which enables batch parallelization. This
precomputation is done through `get_considered_visits_table`.
    

See Deepmind's MCTX for more details
https://github.com/deepmind/mctx/tree/main/mctx/_src
"""
function gumbel_explore(mcts, envs, rng::AbstractRNG)
    tree = create_tree(mcts, envs)
    (; A, B, N) = size(tree)

    gumbel = DeviceArray(mcts.device)(rand(rng, Gumbel(), (A, B)))
    considered_visits_table = DeviceArray(mcts.device)(get_considered_visits_table(mcts, A))

    for simnum in 2:N
        parent_frontier = gumbel_select(mcts, tree, simnum, gumbel, considered_visits_table)
        frontier = eval!(mcts, tree, simnum, parent_frontier)
        backpropagate!(mcts, tree, frontier)
    end
    return tree
end

"""
    get_considered_visits_sequence(max_num_actions, num_simulations)    

Precompute the Gumbel simulations orchestration.

The Gumbel simulations orchestration is done originally iteratively. At each step, a certain
number of considered actions is explored. At the end of exploration, the number of
considered actions is then divided by two. The iterative process is repeated until only two
actions are considered.

Each step has an equal number of simulations so that the total number of simulations is
evenly distributed between them. Likewise, at each step, the number of simulations of the
step is evenly distributed between the most promising considered actions.

This simulation orchestration can be precomputed as in MCTX by saving a sequence
of the considered number of visits for each simulation. In other words, this sequence
indicates for each simulation a constraint on the number of visits (i.e the number of
visits that the selected action at the root node should match)
"""
function get_considered_visits_sequence(max_num_actions, num_simulations)
    (max_num_actions <= 1) && return SVector{num_simulations,Int16}(0:(num_simulations - 1))

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

"""
    get_considered_visits_table(mcts, num_actions)

Return a table containing the precomputed sequence of visits for each number of considered
actions possible.

Sayed in other words, for a given number of considered actions, this table contains a
precomputed sequence. This sequence indicates for each simulation a constraint on the number
of visits (i.e the number of visits that the selected action at the root node should match).

See also [`get_considered_visits_sequence`](@ref)
"""
function get_considered_visits_table(mcts, num_actions)
    ret = imap(1:num_actions) do num_considered_actions
        get_considered_visits_sequence(num_considered_actions, mcts.num_simulations)
    end
    return SVector{num_actions}(ret)
end

"""
    gumbel_select(mcts, tree, simnum, gumbel, considered_visits_table)

Gumbel's variation of classical `select`.

The only difference lies in the use of `gumbel_select_root_action` to select the action at
the root node. `gumbel_select` then fall back on `select` for non-root node.
    
See also [`gumbel_select_root_action`](@ref)
"""
function gumbel_select(mcts, tree, simnum, gumbel, considered_visits_table)
    (; A, N, B) = size(tree)
    tree_size = (Val(A), Val(N), Val(B))

    parent_frontier = zeros(Int16, mcts.device, (2, B))
    Devices.foreach(1:B, mcts.device) do bid
        aid = gumbel_select_root_action(
            mcts, tree, bid, gumbel, considered_visits_table, simnum - ROOT, tree_size
        )
        @assert aid != NO_ACTION

        cnid = tree.children[aid, ROOT, bid]
        new_frontier = if (cnid != UNVISITED)
            select(mcts, tree, bid, tree_size; start=cnid)
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
    get_penality(
        mcts, tree, bid, considered_visits_table, child_visits,
        tree_size::Tuple{Val{A},Any,Any}
    ) where {A}

Computes penalty for actions that do not comply with the constraint on the number of visits.

More precisely, if an action does not respect this constraint, it will have a penalty of
`-Inf32` on its computed score before applying the argmax. This ultimately blocks this action
from being selected if at least another action has not been penalized (which should be the
case).

Actions that comply with the constraint on the number of visits have no penalty (i.e. a
penalty of `0`)
"""
function get_penality(
    mcts, tree, bid, considered_visits_table, child_visits, tree_size::Tuple{Val{A},Any,Any}
) where {A}
    num_valid_actions = sum(aid -> tree.valid_actions[aid, ROOT, bid], 1:A; init=NO_ACTION)
    num_considered_actions = min(mcts.num_considered_actions, num_valid_actions)

    num_visits = get_num_child_visits(tree, ROOT, bid, tree_size)
    considered_visits = considered_visits_table[num_considered_actions][child_visits]
    penality = imap(1:A) do aid
        (num_visits[aid] == considered_visits) ? Float32(0) : -Inf32
    end
    return SVector{A}(penality)
end

"""
    gumbel_select_root_action(
        mcts,
        tree,
        bid,
        gumbel,
        considered_visits_table,
        child_total_visits,
        tree_size::Tuple{Val{A},Any,Any},
    ) where {A}

Gumbel's variation of `select_action` for root node only.

The only difference lies in the use of `gumbel` noise in the computation of the scores
before applying the argmax and the additional constraints on the number of visits.
"""
function gumbel_select_root_action(
    mcts,
    tree,
    bid,
    gumbel,
    considered_visits_table,
    child_visits,
    tree_size::Tuple{Val{A},Any,Any},
) where {A}
    batch_gumbel = SVector{A}(imap(aid -> gumbel[aid, bid], 1:A))
    policy = target_policy(mcts, tree, ROOT, bid, tree_size)
    penality = get_penality(
        mcts, tree, bid, considered_visits_table, child_visits, tree_size
    )

    scores = batch_gumbel + policy + penality
    return Int16(argmax(scores; init=(NO_ACTION, -Inf32)))
end

end