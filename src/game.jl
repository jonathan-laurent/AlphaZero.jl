"""
A generic interface for single-player games and two-player zero-sum games.

Stochastic games and intermediate rewards are supported. By convention,
rewards are expressed from the point of view of the player called _white_.
In two-player zero-sum games, we call `black` the player trying to minimize the reward.
"""
module GameInterface

export AbstractGameSpec, AbstractGameEnv

using ..AlphaZero: Util

#####
##### Game environments and game specifications
#####

# NAMING CONVENTION:
# - A game specification is usually written: `game_spec` or `gspec`.
# - A game environment is usually written: `game` or `g`.
#   We usually avoid `env` as it is too generic and can clash with the name for an
#   MCTS environment or training environment.
# - A state is usually written: `state` or `s`.
# - An action is usually written: `action` or `a`.

"""
    AbstractGameSpec

Abstract type for a game specification.

The specification holds all _static_ information about a game, which does not
depend on the current state.
"""
abstract type AbstractGameSpec end

"""
    AbstractGameEnv

Abstract base type for a game environment.

Intuitively, a game environment holds a game specification and a current state.
"""
abstract type AbstractGameEnv end

"""
    init(::AbstractGameSpec) :: AbstractGameEnv

Create a new game environment in a (possibly random) initial state.
"""
function init end

"""
    spec(game::AbstractGameEnv) :: AbstractGameSpec

Return the game specification of an environment.
"""
function spec end

#####
##### Queries on specs
#####

"""
    two_players(::AbstractGameSpec) :: Bool

Return whether or not a game is a two-players game.
"""
function two_players end

"""
    actions(::AbstractGameSpec)

Return the vector of all game actions.
"""
function actions end

"""
    vectorize_state(::AbstractGameSpec, state) :: Array{Float32}

Return a vectorized representation of a given state.
"""
function vectorize_state end

#####
##### Operations on envs
#####

"""
    set_state!(game::AbstractGameEnv, state)

Modify the state of a game environment in place.
"""
function set_state! end

"""
    current_state(game::AbstractGameEnv)

Return the game state.

!!! warn

    The state returned by this function may be stored (e.g. in the MCTS tree) and must
    therefore either be fresh or persistent. If in doubt, you should make a copy.

"""
function current_state end

# TODO: maybe MCTS should make the copy itself. The performance cost should not be great
# and it would probably avoid people a lot of pain.

"""
    game_terminated(::AbstractGameEnv)

Return a boolean indicating whether or not the game is in a terminal state.
"""
function game_terminated end

"""
    white_playing(::AbstractGameEnv) :: Bool

Return `true` if white is to play and `false` otherwise.

For a one-player game, this function must always return `true`.
"""
function white_playing end

"""
    actions_mask(::AbstractGameEnv)

Return a boolean mask indicating what actions are available.

The following identities must hold:

  - `game_terminated(game) || any(actions_mask(game))`
  - `length(actions_mask(game)) == length(actions(spec(game)))`
"""
function actions_mask end

"""
    play!(game::AbstractGameEnv, action)

Update the game environment by making the current player perform `action`.
Note that this function does not have to be deterministic.
"""
function play! end

"""
    white_reward(game::AbstractGameEnv)

Return the intermediate reward obtained by the white player after the last
transition step. The result is undetermined when called at an initial state.
"""
function white_reward end

"""
    heuristic_value(game::AbstractGameEnv)

Return a heuristic estimate of the state value for the current player.

The given state must be nonfinal and returned values must belong to the
``(-∞, ∞)`` interval.

This function is not needed by AlphaZero but it is useful for building
baselines such as minmax players.
"""
function heuristic_value end

#####
##### Symmetries
#####

"""
    symmetries(::AbstractGameSpec, state)

Return the vector of all pairs `(s, σ)` where:
  - `s` is the image of `state` by a nonidentical symmetry
  - `σ` is the associated actions permutation, as an integer vector of
     size `num_actions(game)`.

A default implementation is provided that returns an empty vector.

Note that the current state of the passed environment is ignored by this function.

# Example

In the game of tic-tac-toe, there are eight symmetries that can be
obtained by composing reflexions and rotations of the board (including the
identity symmetry).

# Property

If `(s2, σ)` is a symmetry for state `s1`, then `mask2 == mask1[σ]` must hold where
`mask1` and `mask2` are the available action masks for `s1` and `s2` respectively.
"""
function symmetries(::AbstractGameSpec, state)
  return Tuple{typeof(state), Vector{Int}}[]
end

#####
##### Interface for interactive exploratory tools
#####

"""
    render(game::AbstractGameEnv)

Print the game state on the standard output.
"""
function render end

"""
    action_string(::AbstractGameSpec, action) :: String

Return a human-readable string representing the provided action.
"""
function action_string end

"""
    parse_action(::AbstractGameSpec, str::String)

Return the action described by string `str` or `nothing` if `str` does not
denote a valid action.
"""
function parse_action end

"""
    read_state(game_spec::AbstractGameSpec)

Read a state from the standard input.
Return the corresponding state (with type `state_type(game_spec)`)
or `nothing` in case of an invalid input.
"""
function read_state end

#####
##### Derived spec functions
#####

"""
    state_type(::AbstractGameSpec)

Return the state type associated to a game.

State objects must be persistent or appear as such as they are stored into
the MCTS tree without copying. They also have to be comparable and hashable.
"""
function state_type(game_spec::AbstractGameSpec)
  return typeof(current_state(init(game_spec)))
end

"""
    state_dim(::AbstractGameSpec)

Return a tuple that indicates the shape of a vectorized state representation.
"""
function state_dim(game_spec::AbstractGameSpec)
  state = current_state(init(game_spec))
  return size(vectorize_state(game_spec, state))
end

"""
    state_memsize(::AbstractGameSpec)

Return the memory footprint occupied by a state of the given game.

The computation is based on a random initial state, assuming that all states have an
identical footprint.
"""
function state_memsize(game_spec::AbstractGameSpec)
  state = current_state(init(game_spec))
  return Base.summarysize(state)
end

"""
    action_type(::AbstractGameSpec)

Return the action type associated to a game.
"""
function action_type(game_spec::AbstractGameSpec)
  return eltype(actions(game_spec))
end

"""
    num_actions(::AbstractGameSpec)

Return the total number of actions associated with a game.
"""
num_actions(game_spec::AbstractGameSpec) = length(actions(game_spec))

"""
    init(::AbstractGameSpec, state) :: AbstractGameEnv

Create a new game environment, initialized in a given state.
"""
function init(gspec::AbstractGameSpec, state)
  env = init(gspec)
  set_state!(env, state)
  return env
end

#####
##### Derived env functions
#####

"""
    clone(::AbstractGameEnv)

Return an independent copy of the given environment.
"""
function clone(game::AbstractGameEnv)
  return init(spec(game), current_state(game))
end

"""
    available_actions(::AbstractGameEnv)

Return the vector of all available actions.
"""
function available_actions(game::AbstractGameEnv)
  mask = actions_mask(game)
  return actions(spec(game))[mask]
end

"""
    apply_random_symmetry!(::AbstractGameEnv)

Update a game environment by applying a random symmetry
to the current state (see [`symmetries`](@ref)).
"""
function apply_random_symmetry!(game::AbstractGameEnv)
  gspec = spec(game)
  syms = symmetries(gspec, current_state(game))
  @assert !isempty(syms) "no symmetries were declared for this game"
  symstate, _ = rand(syms)
  set_state!(game, symstate)
  return
end

end