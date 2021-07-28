"""
Utilities for using AlphaZero.jl on RL environments that implement CommonRLInterface.jl.
"""
module CommonRLInterfaceWrapper

using ..AlphaZero
import CommonRLInterface
using Setfield

const RL = CommonRLInterface

#####
##### Wrappers
#####

mutable struct Env{E, H} <: AbstractGameEnv
  rlenv :: E
  last_reward :: Float64
  two_players :: Bool
  heuristic_value :: H
  vectorize_state :: Function
  symmetries :: Function
  render :: Function
  action_string :: Function
  parse_action :: Function
  read_state :: Function
end

mutable struct Spec{E, H} <: AbstractGameSpec
  env :: Env{E, H}
end

#####
##### Default functions and constructors
#####

function GI.vectorize_state(::RL.AbstractEnv, state)
  @assert isa(state, Array{<:Number}) "
  Your state is not a vector and so you should define `vectorize_state`"
  return convert(Array{Float32}, state)
end

default_vectorize_state(rl, state) = GI.vectorize_state(rl, state)

GI.symmetries(::RL.AbstractEnv, state) = Tuple{typeof(state), Vector{Int}}[]

default_symmetries(rl, state) = GI.symmetries(rl, state)

default_heurisic_value(rl)   = GI.heuristic_value(rl)
default_render(rl)           = GI.render(rl)
default_action_string(rl, a) = GI.action_string(rl, a)
default_parse_action(rl, s)  = GI.parse_action(rl, s)
default_read_state(rl)       = GI.read_state(rl)

"""
    Env(rlenv::CommonRLInterface.AbstractEnv; <kwargs>) <: AbstractGameEnv

Wrap an environment implementing the interface defined in CommonRLInterface.jl into
an `AbstractGameEnv`.

# Requirements

The following optional methods must be implemented for `rlenv`:

  - `clone`
  - `state`
  - `setstate!`
  - `valid_action_mask`
  - `player`
  - `players`

# Keyword arguments

The following optional functions from `GameInterface` are not present in
CommonRLInterface.jl and can be provided as keyword arguments:

  - `vectorize_state`: must be provided unless states already have type `Array{<:Number}`
  - `heuristic_value`
  - `symmetries`
  - `render`
  - `action_string`
  - `parse_action`
  - `read_state`

If `f` is not provided, the default implementation calls
`GI.f(::CommonRLInterface.AbstractEnv, ...)`.
"""
function Env(
    rlenv :: RL.AbstractEnv;
    heuristic_value = default_heurisic_value,
    vectorize_state = default_vectorize_state,
    symmetries      = default_symmetries,
    render          = default_render,
    action_string   = default_action_string,
    parse_action    = default_parse_action,
    read_state      = default_read_state)

  nplayers = length(RL.players(rlenv))
  @assert nplayers == 1 || nplayers == 2 "
  AlphaZero.jl only supports games with one or two players."
  two_players = nplayers == 2

  return Env(
    rlenv, 0., two_players,
    heuristic_value, vectorize_state, symmetries,
    render, action_string, parse_action, read_state)
end

"""
    Spec(rlenv::RL.AbstractEnv; kwargs...) = spec(Env(rlenv; kwargs...))
"""
Spec(rlenv::RL.AbstractEnv; funs...) = Spec(Env(rlenv; funs...))

#####
##### GameInterface API
#####

GI.clone(env::Env) = @set env.rlenv = RL.clone(env.rlenv)

GI.set_state!(env::Env, state) = RL.setstate!(env.rlenv, state)

function GI.init(spec::Spec)
  env = GI.clone(spec.env)
  RL.reset!(env.rlenv)
  return env
end

function GI.init(spec::Spec, state)
  env = GI.clone(spec.env)
  RL.setstate!(env.rlenv, state)
  return env
end

GI.spec(env::Env) = Spec(env)

# Queries on specifications

GI.two_players(spec::Spec) = spec.env.two_players

GI.actions(spec::Spec) = RL.actions(spec.env.rlenv)

GI.vectorize_state(spec::Spec, state) = spec.env.vectorize_state(spec.env.rlenv, state)

# Operations on environments

GI.current_state(env::Env) = RL.state(env.rlenv)

GI.game_terminated(env::Env) = RL.terminated(env.rlenv)

GI.white_playing(env::Env) = RL.player(env.rlenv) == 1

GI.actions_mask(env::Env) = RL.valid_action_mask(env.rlenv)

function GI.play!(env::Env, action)
  r = RL.act!(env.rlenv, action)
  env.last_reward = r
  return
end

GI.white_reward(env::Env) = env.last_reward

GI.heuristic_value(env::Env) = env.heuristic_value(env.rlenv)

# Symmetries

GI.symmetries(spec::Spec, state) = spec.env.symmetries(spec.env.rlenv, state)

# Interactive utilities

GI.render(env::Env) = env.render(env.rlenv)

GI.action_string(spec::Spec, a) = spec.env.action_string(spec.env.rlenv, a)

GI.parse_action(spec::Spec, s) = spec.env.parse_action(spec.env.rlenv, s)

GI.read_state(spec::Spec) = spec.env.read_state(spec.env.rlenv)

end