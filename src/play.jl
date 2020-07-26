#####
##### Interface for players
#####

"""
    AbstractPlayer{Game}

Abstract type for a player of `Game`.
"""
abstract type AbstractPlayer{Game} end

GameType(::AbstractPlayer{Game}) where Game = Game

"""
    think(::AbstractPlayer, game)

Return a probability distribution over actions as a `(actions, π)` pair.
"""
function think end

"""
    reset_player!(::AbstractPlayer)

Reset the internal memory of a player (e.g. the MCTS tree).
The default implementation does nothing.
"""
function reset_player!(::AbstractPlayer)
  return
end

"""
    player_temperature(::AbstractPlayer, game, turn_number)

Return the player temperature, given the number of actions that have
been played before by both players in the current game.

A default implementation is provided that always returns 1.
"""
function player_temperature(::AbstractPlayer, game, turn)
  return 1.0
end

"""
    select_move(player::AbstractPlayer, game, turn_number)

Return a single action. A default implementation is provided that samples
an action according to the distribution computed by [`think`](@ref), with a
temperature given by [`player_temperature`](@ref).
"""
function select_move(player::AbstractPlayer, game, turn_number)
  actions, π = think(player, game)
  τ = player_temperature(player, game, turn_number)
  π = apply_temperature(π, τ)
  return actions[Util.rand_categorical(π)]
end

#####
##### Random Player
#####

"""
    RandomPlayer{Game} <: AbstractPlayer{Game}

A player that picks actions uniformly at random.
"""
struct RandomPlayer{Game} <: AbstractPlayer{Game} end

function think(player::RandomPlayer, game)
  actions = GI.available_actions(game)
  n = length(actions)
  π = ones(n) ./ length(actions)
  return actions, π
end

#####
##### Epsilon-greedy player
#####

"""
    EpsilonGreedyPlayer{Game, Player} <: AbstractPlayer{Game}

A wrapper on a player that makes it choose a random move
with a fixed ``ϵ`` probability.
"""
struct EpsilonGreedyPlayer{G, P} <: AbstractPlayer{G}
  player :: P
  ϵ :: Float64
  function EpsilonGreedyPlayer(p::AbstractPlayer{G}, ϵ) where G
    return new{G, typeof(p)}(p, ϵ)
  end
end

function think(p::EpsilonGreedyPlayer, game)
  actions, π = think(p.player, game)
  n = length(actions)
  η = ones(n) ./ n
  return actions, (1 - p.ϵ) * π + p.ϵ * η
end

function reset!(p::EpsilonGreedyPlayer)
  reset!(p.player)
end

function player_temperature(p::EpsilonGreedyPlayer, game, turn)
  return player_temperature(p.player, game, turn)
end

#####
##### Player with a custom temperature
#####

"""
    PlayerWithTemperature{Game, Player} <: AbstractPlayer{Game}

A wrapper on a player that enables overwriting the temperature schedule.
"""
struct PlayerWithTemperature{G, P} <: AbstractPlayer{G}
  player :: P
  temperature :: AbstractSchedule{Float64}
  function PlayerWithTemperature(p::AbstractPlayer{G}, τ) where G
    return new{G, typeof(p)}(p, τ)
  end
end

function think(p::PlayerWithTemperature, game)
  return think(p.player, game)
end

function reset!(p::PlayerWithTemperature)
  reset!(p.player)
end

function player_temperature(p::PlayerWithTemperature, game, turn)
  return p.temperature[turn]
end

#####
##### An MCTS-based player
#####

"""
    MctsPlayer{Game, MctsEnv} <: AbstractPlayer{Game}

A player that selects actions using MCTS.

# Constructors

    MctsPlayer(mcts::MCTS.Env; τ, niters, timeout=nothing)

Construct a player from an MCTS environment. When computing each move:

- if `timeout` is provided,
  MCTS simulations are executed for `timeout` seconds by groups of `niters`
- otherwise, `niters` MCTS simulations are run

The temperature parameter `τ` can be either a real number or a
[`AbstractSchedule`](@ref).

    MctsPlayer(oracle::MCTS.Oracle, params::MctsParams; timeout=nothing)

Construct an MCTS player from an oracle and an [`MctsParams`](@ref) structure.
"""
struct MctsPlayer{G, M} <: AbstractPlayer{G}
  mcts :: M
  niters :: Int
  timeout :: Union{Float64, Nothing}
  τ :: AbstractSchedule{Float64} # Temperature
  function MctsPlayer(mcts::MCTS.Env{G}; τ, niters, timeout=nothing) where G
    @assert niters > 0
    @assert isnothing(timeout) || timeout > 0
    new{G, typeof(mcts)}(mcts, niters, timeout, τ)
  end
end

# Alternative constructor
function MctsPlayer(
    oracle::MCTS.Oracle{G}, params::MctsParams; timeout=nothing) where G
  mcts = MCTS.Env{G}(oracle,
    gamma=params.gamma,
    cpuct=params.cpuct,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α,
    prior_temperature=params.prior_temperature)
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature,
    timeout=timeout)
end

# MCTS with random oracle
function RandomMctsPlayer(::Type{G}, params::MctsParams) where G
  oracle = MCTS.RandomOracle{G}()
  mcts = MCTS.Env{G}(oracle,
    cpuct=params.cpuct,
    gamma=params.gamma,
    noise_ϵ=params.dirichlet_noise_ϵ,
    noise_α=params.dirichlet_noise_α)
  return MctsPlayer(mcts,
    niters=params.num_iters_per_turn,
    τ=params.temperature)
end

function think(p::MctsPlayer, game)
  if isnothing(p.timeout) # Fixed number of MCTS simulations
    MCTS.explore!(p.mcts, game, p.niters)
  else # Run simulations until timeout
    start = time()
    while time() - start < p.timeout
      MCTS.explore!(p.mcts, game, p.niters)
    end
  end
  return MCTS.policy(p.mcts, game)
end

function player_temperature(p::MctsPlayer, game, turn)
  return p.τ[turn]
end

function reset_player!(player::MctsPlayer)
  MCTS.reset!(player.mcts)
end

#####
##### Network player
#####

"""
    NetworkPlayer{Game, Net} <: AbstractPlayer{Game}

A player that uses the policy output by a neural network directly,
instead of relying on MCTS. The given neural network must be in test mode.
"""
struct NetworkPlayer{G, N} <: AbstractPlayer{G}
  network :: N
   function NetworkPlayer(nn::MCTS.Oracle{G}) where G
    return new{G, typeof(nn)}(nn)
  end
end

function think(p::NetworkPlayer, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  π, _ = p.network(state)
  return actions, π
end

#####
##### Merging two players into one
#####

"""
    TwoPlayers{Game} <: AbstractPlayer{Game}

If `white` and `black` are two [`AbstractPlayer`](@ref), then
`TwoPlayers(white, black)` is a player that behaves as `white` when `white`
is to play and as `black` when `black` is to play.
"""
struct TwoPlayers{G, W, B} <: AbstractPlayer{G}
  white :: W
  black :: B
  function TwoPlayers(white, black)
    G = GameType(white)
    @assert G == GameType(black)
    return new{G, typeof(white), typeof(black)}(white, black)
  end
end

flipped_colors(p::TwoPlayers) = TwoPlayers(p.black, p.white)

function think(p::TwoPlayers, game)
  if GI.white_playing(game)
    return think(p.white, game)
  else
    return think(p.black, game)
  end
end

function select_move(p::TwoPlayers, game, turn)
  if GI.white_playing(game)
    return select_move(p.white, game, turn)
  else
    return select_move(p.black, game, turn)
  end
end

function reset!(p::TwoPlayers)
  reset!(p.white)
  reset!(p.black)
end

function player_temperature(p::TwoPlayers, game, turn)
  if GI.white_playing(game)
    return player_temperature(p.white, game, turn)
  else
    return player_temperature(p.black, game, turn)
  end
end

#####
##### Players can play against each other
#####

"""
    play_game(player; flip_probability=0.) :: Trace

Simulate a game by an [`AbstractPlayer`](@ref) and return a trace.

- For two-player games, please use [`TwoPlayers`](@ref).
- If the `flip_probability` argument is set to ``p``, the board
  is _flipped_ randomly at every turn with probability ``p``,
  using [`GI.apply_random_symmetry`](@ref).
"""
function play_game(player; flip_probability=0.)
  Game = GameType(player)
  game = Game()
  trace = Trace{Game}(GI.current_state(game))
  while true
    if GI.game_terminated(game)
      return trace
    end
    if !iszero(flip_probability) && rand() < flip_probability
      game = GI.apply_random_symmetry(game)
    end
    actions, π_target = think(player, game)
    τ = player_temperature(player, game, length(trace))
    π_sample = apply_temperature(π_target, τ)
    a = actions[Util.rand_categorical(π_sample)]
    GI.play!(game, a)
    push!(trace, π_target, GI.white_reward(game), GI.current_state(game))
  end
end

#####
##### Utilities to manage inference servers
#####

function fill_and_evaluate(net, batch; batch_size, fill)
  n = length(batch)
  @assert n > 0
  if !fill
    return Network.evaluate_batch(net, batch)
  else
    nmissing = batch_size - n
    @assert nmissing >= 0
    if nmissing == 0
      return Network.evaluate_batch(net, batch)
    else
      append!(batch, [batch[1] for _ in 1:nmissing])
      return Network.evaluate_batch(net, batch)[1:n]
    end
  end
end

# Start a server that processes inference requests for TWO networks
# Expected queries must have fields `state` and `netid`.
function inference_server(
    net1::AbstractNetwork,
    net2::AbstractNetwork,
    nworkers;
    fill_batches)
  return Batchifier.launch_server(nworkers) do batch
    n = length(batch)
    mask1 = findall(b -> b.netid == 1, batch)
    mask2 = findall(b -> b.netid == 2, batch)
    @assert length(mask1) + length(mask2) == n
    state(x) = x.state
    batch1 = state.(batch[mask1])
    batch2 = state.(batch[mask2])
    if isempty(mask2) # there are only queries from net1
      return fill_and_evaluate(net1, batch1; batch_size=n, fill=fill_batches)
    elseif isempty(mask1) # there are only queries from net2
      return fill_and_evaluate(net2, batch2; batch_size=n, fill=fill_batches)
    else # both networks sent queries
      res1 = fill_and_evaluate(net1, batch1; batch_size=n, fill=fill_batches)
      res2 = fill_and_evaluate(net2, batch2; batch_size=n, fill=fill_batches)
      @assert typeof(res1) == typeof(res2)
      res = similar(res1, n)
      res[mask1] = res1
      res[mask2] = res2
      return res
    end
  end
end

# Start an inference server for one agent
function inference_server(net::AbstractNetwork, n; fill_batches)
  return Batchifier.launch_server(n) do batch
    fill_and_evaluate(net, batch; batch_size=n, fill=fill_batches)
  end
end

ret_oracle(x) = () -> x
do_nothing!() = nothing
send_done!(reqc) = () -> Batchifier.client_done!(reqc)
zipthunk(f1, f2) = () -> (f1(), f2())

# Two neural network oracles
function batchify_oracles(os::Tuple{AbstractNetwork, AbstractNetwork}, fill, n)
  reqc = inference_server(os[1], os[2], n, fill_batches=fill)
  G = GameType(os[1])
  make1() = Batchifier.BatchedOracle{G}(st -> (state=st, netid=1), reqc)
  make2() = Batchifier.BatchedOracle{G}(st -> (state=st, netid=2), reqc)
  return zipthunk(make1, make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, AbstractNetwork}, fill, n)
  reqc = inference_server(os[2], n, fill_batches=fill)
  make2() = Batchifier.BatchedOracle{GameType(os[2])}(reqc)
  return zipthunk(ret_oracle(os[1]), make2), send_done!(reqc)
end

function batchify_oracles(os::Tuple{AbstractNetwork, <:Any}, fill, n)
  reqc = inference_server(os[1], n, fill_batches=fill)
  make1() = Batchifier.BatchedOracle{GameType(os[1])}(reqc)
  return zipthunk(make1, ret_oracle(os[2])), send_done!(reqc)
end

function batchify_oracles(os::Tuple{<:Any, <:Any}, fill, n)
  return zipthunk(ret_oracle(os[1]), ret_oracle(os[2])), do_nothing!
end

function batchify_oracles(o::AbstractNetwork, fill, n)
  reqc = inference_server(o, n, fill_batches=fill)
  make() = Batchifier.BatchedOracle{GameType(o)}(reqc)
  return make, send_done!(reqc)
end

function batchify_oracles(o::Any, fill, n)
  return ret_oracle(o), do_nothing!
end

#####
##### Distributed simulator
#####

"""
    Simulator(make_player, oracles, measure)

A distributed simulator that encapsulates the details of running simulations
across multiple threads and multiple machines.

# Arguments

    - `make_oracles`: a function that takes no argument and returns
       the oracles used by the player, which can be either
      `nothing`, a single oracle or a pair of oracles.
    - `make_player`: a function that takes as an argument the `oracles` field
      above and nuild a player from it.
    - `measure(trace, colors_flipped, player)`: the function that is used to
      take measurements after each game simulation.
"""
struct Simulator{MakePlayer, Oracles, Measure}
  make_player :: MakePlayer
  make_oracles :: Oracles
  measure :: Measure
end

"""
    record_trace

A measurement function to be passed to a [`Simulator`](@ref) that produces
named tuples with two fields: `trace::Trace` and `colors_flipped::Bool`.
"""
record_trace(t, cf, p) = (trace=t, colors_flipped=cf)

"""
    @enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

Policy for attributing colors in a duel between a baseline and a contender.
"""
@enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE


"""
    simulate(::Simulator; <keyword arguments>)

Play a series of games using a given [`Simulator`](@ref).

# Keyword Arguments

  - `num_games`: number of games to play
  - `num_workers`: number of workers tasks to spawn
  - `game_simulated`: called every time a game simulation is completed
  - `reset_every`: if set, players are reset every `reset_every` games
  - `color_policy`: either `nothing` or a [`ColorPolicy`](@ref)
  - `flip_probability=0.`: see [`play_game`](@ref)

# Return

Return a vector of objects returned by `simulator.measure`.
"""
function simulate(
    simulator::Simulator;
    num_games,
    num_workers,
    game_simulated,
    reset_every=nothing,
    fill_batches=true,
    flip_probability=0.,
    color_policy=nothing)

  oracles = simulator.make_oracles()
  spawn_oracles, done =
    batchify_oracles(oracles, fill_batches, num_workers)
  return Util.mapreduce(1:num_games, num_workers, vcat, []) do
    oracles = spawn_oracles()
    player = simulator.make_player(oracles)
    worker_sim_id = 0
    # For each worker
    function simulate_game(sim_id)
      worker_sim_id += 1
      # Switch players' colors if necessary
      if !isnothing(color_policy)
        @assert isa(player, TwoPlayers)
        colors_flipped =
          (color_policy == BASELINE_WHITE) ||
          (color_policy == ALTERNATE_COLORS && sim_id % 2 == 1)
        # "_pf" stands for "possibly flipped"
        player_pf = colors_flipped ? flipped_colors(player) : player
      else
        colors_flipped = false
        player_pf = player
      end
      # Play the game and generate a report
      trace = play_game(player_pf, flip_probability=flip_probability)
      report = simulator.measure(trace, colors_flipped, player)
      # Reset the player periodically
      if !isnothing(reset_every) && worker_sim_id % reset_every == 0
        reset_player!(player)
      end
      # Signal that a game has been simulated
      game_simulated()
      return report
    end
    return (process=simulate_game, terminate=done)
  end
end

"""
    simulate_distributed(::Simulator; <keyword arguments>)

Identical to [`simulate`](@ref) but splits the work
across all available workers.
"""
function simulate_distributed(
    simulator::Simulator;
    num_games,
    num_workers,
    game_simulated,
    reset_every=nothing,
    fill_batches=true,
    flip_probability=0.,
    color_policy=nothing)

  # Spawning a task to keep count of completed simulations
  chan = Distributed.RemoteChannel(()->Channel{Nothing}(1))
  Threads.@spawn begin
    for i in 1:num_games
      take!(chan)
      game_simulated()
    end
  end
  remote_game_simulated() = put!(chan, nothing)
  # Distributing the simulations across workers
  num_each, rem = divrem(num_games, Distributed.nworkers())
  @assert num_each >= 1
  workers = Distributed.workers()
  tasks = map(workers) do w
    Distributed.@spawnat w begin
      Util.@printing_errors begin
        simulate(
          simulator,
          num_games=(w == workers[1] ? num_each + rem : num_each),
          num_workers=num_workers,
          game_simulated=remote_game_simulated,
          reset_every=reset_every,
          fill_batches=fill_batches,
          flip_probability=flip_probability,
          color_policy=color_policy)
        end
    end
  end
  results = fetch.(tasks)
  # If one of the worker raised an exception, we print it
  for r in results
    if isa(r, Distributed.RemoteException)
      showerror(stderr, r, catch_backtrace())
    end
  end
  return reduce(vcat, results)
end

function compute_redundancy(states)
  unique = Set(states)
  return 1. - length(unique) / length(states)
end

# samples is a vector of named tuples with fields `trace` and `colors_flipped`
function rewards_and_redundancy(samples; gamma)
  rewards = map(samples) do s
    wr = total_reward(s.trace, gamma)
    return s.colors_flipped ? -wr : wr
  end
  states = [st for s in samples for st in s.trace.states]
  redundancy = compute_redundancy(states)
  return rewards, redundancy
end

#####
##### Simple utilities to play an interactive game
#####

"""
    Human{Game} <: AbstractPlayer{Game}

Human player that queries the standard input for actions.

Does not implement [`think`](@ref) but instead implements
[`select_move`](@ref) directly.
"""
struct Human{Game} <: AbstractPlayer{Game} end

struct Quit <: Exception end

function select_move(::Human, game, turn)
  a = nothing
  while isnothing(a) || a ∉ GI.available_actions(game)
    print("> ")
    str = readline()
    print("\n")
    isempty(str) && throw(Quit())
    a = GI.parse_action(game, str)
  end
  return a
end

"""
    interactive!(game, white, black)

Launch an interactive session for `game::AbstractGame` between players
`white` and `black`. Both players have type `AbstractPlayer` and one of them
is typically [`Human`](@ref).
"""
function interactive!(game, player)
  try
  GI.render(game)
  turn = 0
  while !GI.game_terminated(game)
    action = select_move(player, game, turn)
    GI.play!(game, action)
    GI.render(game)
    turn += 1
  end
  catch e
    isa(e, Quit) || rethrow(e)
    return
  end
end

interactive!(game, white, black) = interactive!(game, TwoPlayers(white, black))

interactive!(game::G) where G = interactive!(game, Human{G}(), Human{G}())
