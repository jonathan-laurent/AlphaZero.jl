#####
##### GameInterface Tests
#####

using AlphaZero

function generate_states(game, n)
  traces = []
  player = RandomPlayer()
  for i in 1:n
    trace = play_game(game, player)
    push!(traces, trace)
  end
  return Set(s for t in traces for s in t.states)
end

same_state(env1, env2) = GI.current_state(env1) == GI.current_state(env2)

function test_game(game, n=100)
  @assert isa(GI.two_players(game), Bool)
  # Testing initial state
  if all(same_state(GI.new_env(game), GI.new_env(game)) for _ in 1:n)
    @info "There appears to be a unique initial state."
  else
    @info "There appears to be a nontrivial distribution of initial states."
  end
  @assert all(GI.white_playing(GI.new_env(game)) for _ in 1:n)  
  # Save the game static properties
  Action = GI.action_type(game)
  State = GI.state_type(game)
  actions = GI.actions(game)
  num_actions = GI.num_actions(game)
  state_dim = GI.state_dim(game)
  two_players = GI.two_players(game)
  # Check properties on random states
  for state in generate_states(game, n)
    GI.reset!(game, state)
    @assert GI.current_state(game) == state
    @assert isa(GI.white_playing(game), Bool)
    @assert isa(GI.white_playing(game), Bool) || GI.two_players(game)
    @assert isa(GI.current_state(game), State)
    @assert GI.current_state(game) == state
    @assert same_state(GI.clone(game), GI.new_env(game, state))
    @assert isa(GI.actions(game), Vector{Action})
    @assert !isempty(GI.actions(game))
    @assert isa(GI.game_terminated(game), Bool)
    vect = GI.vectorize_state(game, state)
    @assert isa(vect, Array{Float32})
    @assert size(vect) == GI.state_dim(game)
    @assert GI.game_terminated(game) || !isempty(GI.available_actions(game))
    @assert length(GI.actions_mask(game)) == length(GI.actions(typeof(game)))
    if !GI.game_terminated(game)
      @assert isa(GI.heuristic_value(game), Float64)
      state_copy = deepcopy(state)
      a = rand(GI.available_actions(game))
      @assert isa(GI.action_string(game, a), String)
      GI.play!(game, a)
      @assert state_copy == state "States must appear as persistent."
      @assert isa(GI.white_reward(game), Float64)
    end
    # Testing symmetries
    syms = GI.symmetries(game, state)
    @assert isa(syms, Vector)
    for sym in syms
      s, σ = sym
      @assert isa(s, State)
      @assert isa(σ, Vector{Int})
      @assert length(σ) == num_actions
      @assert GI.test_symmetry(game, sym)
    end
    # Verifying that static properties are state invariant
    @assert GI.num_actions(game) == num_actions
    @assert GI.actions(game) == actions
    @assert GI.state_type(game) == State
    @assert GI.action_type(game) == State
    @assert GI.two_players(game) == two_players
    @assert GI.state_dim(game) == state_dim
  end
end
