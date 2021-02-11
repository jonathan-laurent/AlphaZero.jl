#####
##### GameInterface Tests
#####

# Generate a set of reachable states randomly
function generate_states(gspec, n)
  traces = [play_game(gspec, RandomPlayer()) for i in 1:n]
  return Set(s for t in traces for s in t.states)
end

same_state(game1, game2) = GI.current_state(game1) == GI.current_state(game2)

# Test that the same (symmetric) actions are still available in a symmetric state
# function test_symmetry(gspec, state, (symstate, aperm))
#   mask = GI.actions_mask(GI.init(gspec, state))
#   symmask = GI.actions_mask(GI.init(gspec, symstate))
#   v = falses(length(symmask))
#   v[mask] .= true
#   v = v[aperm]
#   return all(v[symmask]) && !any(v[.~symmask])
# end

function test_symmetry(gspec, state, (symstate, aperm))
  mask = GI.actions_mask(GI.init(gspec, state))
  symmask = GI.actions_mask(GI.init(gspec, symstate))
  return symmask == mask[aperm]
end

"""

    test_game(::AbstractGameSpec)
    test_game(experiment)

Perform some sanity checks regarding the compliance of a game with the
AlphaZero.jl [Game Interface](@ref game_interface).
"""
function test_game(gspec::AbstractGameSpec; n=100)
  @assert isa(GI.two_players(gspec), Bool)

  # Testing initial states
  if all(same_state(GI.init(gspec), GI.init(gspec)) for _ in 1:n)
    @info "There appears to be a unique initial state."
  else
    @info "There appears to be a nontrivial distribution of initial states."
  end
  @assert all(GI.white_playing(GI.init(gspec)) for _ in 1:n)  

  # Save the game static properties
  State = GI.state_type(gspec)
  state_dim = GI.state_dim(gspec)
  Action = GI.action_type(gspec)
  actions = GI.actions(gspec)
  @assert isa(actions, AbstractVector{Action})
  @assert !isempty(actions)
  num_actions = GI.num_actions(gspec)
  two_players = GI.two_players(gspec)

  # Check properties on random states
  for state in generate_states(gspec, n)
    game = GI.init(gspec, state)
    @assert GI.current_state(game) == state
    GI.set_state!(game, state)
    @assert GI.current_state(game) == state
    @assert isa(GI.current_state(game), State)
    @assert same_state(GI.clone(game), GI.init(gspec, state))

    @assert isa(GI.white_playing(game), Bool)
    @assert GI.white_playing(game) || two_players
    
    @assert isa(GI.game_terminated(game), Bool)
    @assert GI.game_terminated(game) || !isempty(GI.available_actions(game))

    @assert eltype(GI.actions_mask(game)) == Bool
    @assert length(GI.actions_mask(game)) == num_actions

    vect = GI.vectorize_state(gspec, state)
    @assert isa(vect, Array{Float32})
    @assert size(vect) == state_dim

    # Playing a move
    if !GI.game_terminated(game)
      @assert isa(GI.heuristic_value(game), Float64)
      state_copy = deepcopy(state)
      a = rand(GI.available_actions(game))
      @assert isa(GI.action_string(gspec, a), String)
      GI.play!(game, a)
      @assert state_copy == state "States must appear as persistent."
      @assert isa(GI.white_reward(game), Float64)
    end

    # Testing symmetries
    syms = GI.symmetries(gspec, state)
    @assert isa(syms, Vector)
    for sym in syms
      s, σ = sym
      @assert isa(s, State)
      @assert isa(σ, Vector{Int})
      @assert length(σ) == num_actions
      @assert test_symmetry(gspec, state, sym)
    end

    # Verifying that static properties are state invariant
    @assert GI.num_actions(GI.spec(game)) == num_actions
    @assert GI.actions(GI.spec(game)) == actions
    @assert GI.state_type(GI.spec(game)) == State
    @assert GI.action_type(GI.spec(game)) == Action
    @assert GI.two_players(GI.spec(game)) == two_players
    @assert GI.state_dim(GI.spec(game)) == state_dim
  end
end
