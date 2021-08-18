module OpenSpielWrapper

using ..AlphaZero
using OpenSpiel


struct Spec{G} <: GI.AbstractGameSpec
  spiel_game :: G # loaded with OpenSpiel.load_game
  suppress_warnings :: Bool
  heuristic_value :: Function
  vectorize_state :: Function
  symmetries :: Function
  action_string :: Function
  parse_action :: Function
  read_state :: Function
end

mutable struct Env{G, S, P} <: GI.AbstractGameEnv
  spec :: Spec{G}
  state :: S
  curplayer :: P
end


#####
##### Default functions and constructors
#####


default_heurisic_value(env) = 0.0
# not handling terminal states, not flip board for two-player games
default_vectorize_state(spec, state) = convert(Array{Float32}, observation_tensor(state))
default_symmetries(spec, state) = GI.symmetries(spec, state)
default_action_string(spec, action) = GI.action_string(spec, action)
default_parse_action(spec, str::String) = GI.parse_action(spec, str)
default_read_state(spec) = GI.read_state(spec)

"""
    Spec(spiel_game; kwargs...) <: AbstractGameSpec


Wrap an OpenSpiel game object (`CxxWrap.StdLib.SharedPtrAllocated{OpenSpiel.Game}`)
into an AlphaZero game spec.

# Keyword arguments

The following optional functions from `GameInterface` are not present in
OpenSpiel.jl and can be provided as keyword arguments:

  - `heuristic_value`
  - `vectorize_state`
  - `symmetries`
  - `action_string`
  - `parse_action`
  - `read_state`

You can silence warnings by setting `suppress_warnings=true`.
"""

function Spec(
  spiel_game;
  suppress_warnings = false,
  heuristic_value = default_heurisic_value,
  vectorize_state = default_vectorize_state,
  symmetries = default_symmetries,
  action_string = default_action_string,
  parse_action = default_parse_action,
  read_state = default_read_state
)
  nplayers = num_players(spiel_game)
  @assert nplayers == 1 || nplayers == 2 "
  AlphaZero.jl only supports games with one or two players."
  if heuristic_value == default_heurisic_value && !suppress_warnings
    @warn "The `default_heuristic_value` function was not provided and so " *
      "algorithms such as MinMax may not work with this environment."
  end
  return Spec(
    spiel_game,
    suppress_warnings,
    heuristic_value,
    vectorize_state,
    symmetries,
    action_string,
    parse_action,
    read_state
  )
end


#####
##### GameInterface API
#####


function GI.init(spec::Spec)
  state = new_initial_state(spec.spiel_game)
  curplayer = current_player(state)
  return Env(spec, state, curplayer)
end

GI.spec(env::Env) = env.spec

# Queries on specifications

GI.two_players(spec::Spec) = num_players(spec.spiel_game) == 2

# TODO kinda ugly, may be game relevant
GI.actions(spec::Spec) = collect(0:num_distinct_actions(spec.spiel_game)-1)

GI.vectorize_state(spec::Spec, state) = spec.vectorize_state(spec, state)

# Operations on environments

GI.current_state(env::Env) = deepcopy(env.state)

function GI.set_state!(env::Env, state) 
  env.state = deepcopy(state)
  env.curplayer = current_player(state)
end


GI.game_terminated(env::Env) = is_terminal(env.state)

function GI.white_playing(env::Env)
  wp = env.curplayer==0
  return is_terminal(env.state) && GI.two_players(env.spec) ? !wp : wp
end

# not all games implement legal_actions_mask
function GI.actions_mask(env::Env)
  try
    return convert(Array{Bool}, legal_actions_mask(env.state))
  catch e
    @error "AlphaZero.jl only works with OpenSpiel environments that implement " * 
      "the `legal_actions_mask` function."
    rethrow(e)
  end
end

#TODO is_chance_node, is_simultaneous_node
function GI.play!(env::Env, action) 
  apply_action(env.state, action)
  if !is_terminal(env.state)
    env.curplayer = current_player(env.state)
  end
end

GI.white_reward(env::Env) = returns(env.state)[1]

# available_actions dispatched
GI.available_actions(env::Env) = legal_actions(env.state)

GI.heuristic_value(env::Env) = env.spec.heuristic_value(env)

# Symmetries

GI.symmetries(spec, state) = spec.symmetries(spec.spiel_game, state)

# Interactive utilities

function GI.render(env::Env)
  show(env.state)
  print("\n\n") # GameInterface expects the rendering to end with an empty line 
end

# action_string, parse action require Env instead of Spec for OpenSpiel functions
GI.action_string(env::Env, action) = action_to_string(env.state, action)
GI.action_string(spec::Spec, action) = spec.action_string(spec, action)

GI.parse_action(env::Env, str::String) = string_to_action(env.state, str)
GI.parse_action(spec::Spec, str::String) = spec.parse_action(spec, str)

GI.read_state(spec::Spec) = spec.read_state(spec.spiel_game)

end # module