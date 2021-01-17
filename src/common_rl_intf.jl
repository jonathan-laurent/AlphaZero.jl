"""
Enable using AlphaZero.jl on RL environments that implement CommonRLInterface.jl.

# Requirements

In addition to the mandatory functions, this bridge relies on the following
optional functions: `clone`, `state`, `setstate!` and `valid_action_mask`.
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

function default_vectorize_state(rl, state)
  err = "Your state is not a vector and so you should define `vectorize_state`"
  @assert isa(state, Array{<:Number}) err
  return convert(Array{Float32}, state)
end

default_symmetries(rl, state) = Tuple{typeof(state), Vector{Int}}[]

default_heurisic_value(rl)   = error("Undefined: heuristic_value")
default_render(rl)           = error("Undefined: render")
default_action_string(rl, a) = error("Undefined: action_string")
default_parse_action(rl, s)  = error("Undefined: parse_action")
default_read_state(rl)       = error("Undefined: read_state")

function Env(
    rlenv :: RL.AbstractEnv;
    heuristic_value = default_heurisic_value,
    vectorize_state = default_vectorize_state,
    symmetries      = default_symmetries,
    render          = default_render,
    action_string   = default_action_string,
    parse_action    = default_parse_action,
    read_state      = default_read_state)

  # TODO: how to compute the number of players with CommonRLInterface?
  #nplayers = length(RL.player_indices(rl))
  #@assert nplayers == 1 || nplayers == 2
  #two_players = nplayers == 2
  two_players = false

  return Env(
    rlenv, 0., two_players,
    heuristic_value, vectorize_state, symmetries,
    render, action_string, parse_action, read_state)
end

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

GI.heuristic_value(env) = env.heuristic_value(env.rlenv)

# Symmetries

GI.symmetries(spec::Spec, state) = spec.env.symmetries(spec.env.rlenv, state)

# Interactive utilities

GI.render(env::Env) = env.render(env.rlenv)

GI.action_string(spec::Spec, a) = spec.env.action_string(spec.env.rlenv, a)

GI.parse_action(spec::Spec, s) = spec.env.parse_action(spec.env.rlenv, s)

GI.read_state(spec::Spec) = spec.env.read_state(spec.env.rlenv)

end