#####
##### GameInterface Tests
#####

using AlphaZero

function generate_states(::Type{Game}, n) where Game
  traces = []
  player = RandomPlayer{Game}()
  for i in 1:n
    trace = play_game(player)
    push!(traces, trace)
  end
  return Set(s for t in traces for s in t.states)
end

same_state(env1, env2) = GI.current_state(env1) ==  GI.current_state(env2)

function test(Game, n=100)
  # Testing initial state
  if all(same_state(Game(), Game()) for _ in 1:n)
    @info "There appears to be a unique initial state."
  else
    @info "There appears to be a nontrivial distribution of initial states."
  end
  @assert all(GI.white_playing(Game()) for _ in 1:n)
  deterministic = true
  for state in generate_states(Game, n)
    game = Game(state)
    @assert isa(game, Game)
    @assert isa(GI.current_state(game), GI.State(Game))
    @assert GI.current_state(game) == state
    @assert same_state(copy(game), Game(state))
    @assert isa(GI.actions(Game), Vector{GI.Action(Game)})
    @assert !isempty(GI.actions(Game))
    @assert isa(GI.game_terminated(game), Bool)
    vect = GI.vectorize_state(Game, state)
    @assert isa(vect, Array{Float32})
    @assert size(vect) == GI.state_dim(Game)
    @assert GI.game_terminated(game) || !isempty(GI.available_actions(game))
    @assert length(GI.actions_mask(game)) == length(GI.actions(typeof(game)))
    if !GI.game_terminated(game)
      @assert isa(GI.heuristic_value(game), Float64)
      state_copy = deepcopy(state)
      a = rand(GI.available_actions(game))
      @assert isa(GI.action_string(Game, a), String)
      GI.play!(game, a)
      @assert state_copy == state "States must appear as persistent."
      @assert isa(GI.white_reward(game), Float64)
    end
    # Testing symmetries
    syms = GI.symmetries(Game, state)
    @assert isa(syms, Vector)
    for sym in syms
      s, σ = sym
      @assert isa(s, GI.State(Game))
      @assert isa(σ, Vector{Int})
      @assert length(σ) == GI.num_actions(Game)
      @assert GI.test_symmetry(Game, state, sym)
    end
  end
  if deterministic
    @info "The transition function appears to be deterministic."
  else
    @info "The transition function appears to be stochastic."
  end
end

#####
##### Main
#####

using Juno

include("../games/tictactoe/main.jl")
include("../games/connect-four/main.jl")

test(Tictactoe.Game, 20)
test(ConnectFour.Game, 20)

#interactive!(ConnectFour.Game())
