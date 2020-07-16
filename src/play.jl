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
   function NetworkPlayer(nn::AbstractNetwork{G}) where G
    return new{G, typeof(nn)}(nn)
  end
end

function think(p::NetworkPlayer, game)
  actions = GI.available_actions(game)
  state = GI.current_state(game)
  π, _ = MCTS.evaluate(p.network, state)
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

function fill_and_evaluate(net, batch; batch_size, fill=false)
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
    n;
    fill_batches=false)

  return Batchifier.launch_server(n) do batch
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
function inference_server(net::AbstractNetwork, n)
  return Batchifier.launch_server(n) do batch
    return Network.evaluate_batch(net, batch)
  end
end

ret_oracle(x) = () -> x
do_nothing!() = nothing
send_done!(reqc) = () -> Batchifier.client_done!(reqc)

# Two neural network oracles
function batchify_oracles(o1::AbstractNetwork, o2::AbstractNetwork, n)
  reqc = inference_server(o1, o2, n, fill_batches=true)
  G = GameType(o1)
  make1() = Batchifier.BatchedOracle{G}(st -> (state=st, netid=1), reqc)
  make2() = Batchifier.BatchedOracle{G}(st -> (state=st, netid=2), reqc)
  return make1, make2, send_done!(reqc)
end

# One neural network oracle
function batchify_oracles(o1, o2::AbstractNetwork, n)
  reqc = inference_server(o2, n, fill_batches=true)
  make2() = Batchifier.BatchedOracle{GameType(o2)}(reqc)
  return ret_oracle(o1), make2, send_done!(reqc)
end

# One neural network oracle (symmetric version)
function batchify_oracles(o1::AbstractNetwork, o2, n)
  reqc = inference_server(o1, n, fill_batches=true)
  make1() = Batchifier.BatchedOracle{GameType(o1)}(reqc)
  return make1, ret_oracle(o2), send_done!(reqc)
end

# No neural network oracle
function batchify_oracles(o1, o2, n)
  return ret_oracle(o1), ret_oracle(o2), do_nothing!
end

#####
##### Pitting two players against each other
#####

"""
    @enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE

Policy for attributing colors in a duel between a baseline and a contender.
"""
@enum ColorPolicy ALTERNATE_COLORS BASELINE_WHITE CONTENDER_WHITE


"""
    pit_players(<keyword arguments>)

Evaluate two `AbstractPlayer` against each other in a series of games.

Return a vector of `(trace::Trace, baseline_white::Bool)` named tuples.

This function can only be used with two-player games.

# Keyword Arguments

  - `make_contender`: function that builds a contender player from an oracle
  - `contender_oracle`: contender oracle, or `nothing`
  - `make_baseline`: function that builds a baseline player from an oracle
  - `num_games`: number of games to play
  - `num_workers`: number of workers tasks to spawn
  - `handler`: called every time a game is played with the simulated trace
  - `reset_every`: if set, players are reset every `reset_every` games
  - `color_policy`: determines the [`ColorPolicy`](@ref),
     which is `ALTERNATE_COLORS` by default
  - `flip_probability=0.`: see [`play_game`](@ref)
"""
function pit_players(;
    make_contender,
    contender_oracle,
    make_baseline,
    baseline_oracle,
    num_games,
    num_workers,
    handler,
    reset_every=nothing,
    flip_probability=0.,
    color_policy=ALTERNATE_COLORS)

  # Naming convention: *_c stands for contender, *_b for baseline
  @assert num_workers <= num_games
  @assert GI.two_players(GameType(contender_oracle))
  lock = ReentrantLock() # only used to surround the calls to `handler`
  make_c, make_b, done =
    batchify_oracles(contender_oracle, baseline_oracle, num_workers)
  res = Util.threads_pmap(1:num_workers) do _
    oracle_c = make_c()
    oracle_b = make_b()
    num_sims = num_games ÷ num_workers
    player_c = make_contender(oracle_c)
    player_b = make_baseline(oracle_b)
    # For each worker
    res = map(1:num_sims) do i
      baseline_white =
        (color_policy == BASELINE_WHITE) ||
        (color_policy == ALTERNATE_COLORS && i % 2 == 1)
      white = baseline_white ? player_b : player_c
      black = baseline_white ? player_c : player_b
      trace = play_game(
        TwoPlayers(white, black),
        flip_probability=flip_probability)
      if !isnothing(reset_every) && i % reset_every == 0
        reset_player!(player_b)
        reset_player!(player_c)
      end
      Base.lock(lock)
      handler(trace)
      Base.unlock(lock)
      return (trace=trace, baseline_white=baseline_white)
    end
    done()
    return res
  end
  return res |> Iterators.flatten |> collect
end

function compute_redundancy(states)
  unique = Set(states)
  return 1. - length(unique) / length(states)
end

# To be called on the output of `pit_players`
function average_reward_and_redundancy(samples; gamma)
  rewards = map(samples) do s
    wr = total_reward(s.trace, gamma)
    return s.baseline_white ? -wr : wr
  end
  avgr = mean(rewards)
  states = [st for s in samples for st in s.trace.states]
  redundancy = compute_redundancy(states)
  return avgr, redundancy
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
