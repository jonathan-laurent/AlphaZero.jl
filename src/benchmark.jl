"""
Utilities to evaluate players against one another.

Typically, between each training iteration, different players
that possibly depend on the current neural network
compete against a set of baselines.
"""
module Benchmark

import ..Network, ..MinMax, ..GI
import ..Env, ..MCTS, ..MctsParams, ..TwoPlayers
import ..simulate, ..Simulator, ..rewards_and_redundancy, ..record_trace
import ..ColorPolicy, ..ALTERNATE_COLORS
import ..AbstractPlayer, ..EpsilonGreedyPlayer, ..NetworkPlayer, ..MctsPlayer
import ..PlayerWithTemperature, ..ConstSchedule

using ProgressMeter
using Statistics: mean

"""
    Benchmark.DuelOutcome

The outcome of a duel between two players.

# Fields
- `player` and `baseline` are `String` fields containing the names of
    both players involved in the duel
- `avgr` is the averagereward collected by `player`
- `rewards` is the sequence of rewards collected by `player` (one per game)
- `redundancy` is the ratio of duplicate positions encountered during the
   evaluation, not counting the initial position. If this number is too high,
   you may want to increase the move selection temperature.
- `time` is the computing time spent running the duel, in seconds
"""
struct DuelOutcome
  player :: String
  baseline :: String
  avgr :: Float64
  rewards :: Vector{Float64}
  redundancy :: Float64
  time :: Float64
end

"""
    Benchmark.Report = Vector{Benchmark.DuelOutcome}

A benchmark report is a vector of [`Benchmark.DuelOutcome`](@ref) objects.
"""
const Report = Vector{DuelOutcome}

"""
    Benchmark.Player

Abstract type to specify a player that can be featured in a benchmark duel.

Subtypes must implement the following functions:
  - `Benchmark.instantiate(player, nn)`: instantiate the player specification
      into an [`AbstractPlayer`](@ref) given a neural network
  - `Benchmark.name(player)`: return a `String` describing the player
"""
abstract type Player end

# instantiate(::Player, nn)
function instantiate end

# name(::Player) :: String
function name end

"""
    Benchmark.Duel(player, baseline; num_games)

Specify a duel that consists in `num_games` games between
`player` and `baseline`, each of them of type [`Benchmark.Player`](@ref).

# Optional keyword arguments
- `reset_every`: if set, the MCTS tree is reset every `reset_mcts_every` games
    to avoid running out of memory
- `color_policy` has type [`ColorPolicy`](@ref) and is `ALTERNATE_COLORS`
    by default
"""
struct Duel
  num_games :: Int
  num_workers :: Int
  use_gpu :: Bool
  reset_every :: Union{Nothing, Int}
  flip_probability :: Float64
  color_policy :: ColorPolicy
  player :: Player
  baseline :: Player
  function Duel(player, baseline;
      num_games, num_workers, use_gpu=false, reset_every=nothing,
      color_policy=ALTERNATE_COLORS, flip_probability=0.)
    return new(num_games, num_workers, use_gpu, reset_every,
      flip_probability, color_policy, player, baseline)
  end
end

"""
    Benchmark.run(env::Env, duel::Benchmark.Duel, progress=nothing)

Run a benchmark duel and return a [`Benchmark.DuelOutcome`](@ref).

If a `progress` is provided, `next!(progress)` is called
after each simulated game.
"""
function run(env::Env{G}, duel::Duel, progress=nothing) where G
  net() = Network.copy(env.bestnn, on_gpu=duel.use_gpu, test_mode=true)
  simulator = Simulator(net, record_trace) do net
    player = instantiate(duel.player, net)
    baseline = instantiate(duel.baseline, net)
    return TwoPlayers(player, baseline)
  end
  samples, elapsed = @timed simulate(
    simulator,
    num_games=duel.num_games,
    num_workers=duel.num_workers,
    game_simulated=(() -> next!(progress)),
    reset_every=duel.reset_every,
    flip_probability=duel.flip_probability,
    color_policy=duel.color_policy)
  gamma = env.params.self_play.mcts.gamma
  rewards, redundancy = rewards_and_redundancy(samples, gamma=gamma)
  avgr = mean(rewards)
  return DuelOutcome(
    name(duel.player), name(duel.baseline), avgr, rewards, redundancy, elapsed)
end

#####
##### Statistics for games with ternary rewards
#####

struct TernaryOutcomeStatistics
  num_won  :: Int
  num_draw :: Int
  num_lost :: Int
end

function TernaryOutcomeStatistics(rewards::AbstractVector{<:Number})
  num_won  = count(==(1), rewards)
  num_draw = count(==(0), rewards)
  num_lost = count(==(-1), rewards)
  @assert num_won + num_draw + num_lost == length(rewards)
  return TernaryOutcomeStatistics(num_won, num_draw, num_lost)
end

function TernaryOutcomeStatistics(outcome::DuelOutcome)
  return TernaryOutcomeStatistics(outcome.rewards)
end

#####
##### Standard players
#####

"""
    Benchmark.MctsRollouts(params) <: Benchmark.Player

Pure MCTS baseline that uses rollouts to evaluate new positions.

Argument `params` has type [`MctsParams`](@ref).
"""
struct MctsRollouts <: Player
  params :: MctsParams
end

name(p::MctsRollouts) = "MCTS ($(p.params.num_iters_per_turn) rollouts)"

function instantiate(p::MctsRollouts, nn::MCTS.Oracle{G}) where G
  return MctsPlayer(MCTS.RolloutOracle{G}(), p.params)
end

"""
    Benchmark.Full(params) <: Benchmark.Player

Full AlphaZero player that combines MCTS with the learnt network.

Argument `params` has type [`MctsParams`](@ref).
"""
struct Full <: Player
  params :: MctsParams
  Full(params) = new(params)
end

name(::Full) = "AlphaZero"

function instantiate(p::Full, nn)
  return MctsPlayer(nn, p.params)
end

"""
    Benchmark.NetworkOnly(;τ=1.0) <: Benchmark.Player

Player that uses the policy output by the learnt network directly,
instead of relying on MCTS.
"""
struct NetworkOnly <: Player
  τ :: Float64
  NetworkOnly(;τ=1.0) = new(τ)
end

name(::NetworkOnly) = "Network Only"

function instantiate(p::NetworkOnly, nn)
  player = NetworkPlayer(nn)
  return PlayerWithTemperature(player, ConstSchedule(p.τ))
end

"""
    Benchmark.MinMaxTS(;depth, τ=0.) <: Benchmark.Player

Minmax baseline, which relies on [`MinMax.Player`](@ref).
"""
struct MinMaxTS <: Player
  depth :: Int
  amplify_rewards :: Bool
  τ :: Float64
  MinMaxTS(;depth, amplify_rewards, τ=0.) = new(depth, amplify_rewards, τ)
end

name(p::MinMaxTS) = "MinMax (depth $(p.depth))"

function instantiate(p::MinMaxTS, ::MCTS.Oracle{G}) where G
  return MinMax.Player{G}(
    depth=p.depth, amplify_rewards=p.amplify_rewards, τ=p.τ)
end

"""
    Benchmark.Solver(;ϵ) <: Benchmark.Player

Perfect solver that plays randomly with probability `ϵ`.
"""
struct Solver <: Player
  ϵ :: Float64
  Solver(;ϵ) = new(ϵ)
end

# Return the type of the perfect player for a given type of game
# PerfectPlayer(::Type{<:GI.AbstractGame})
function PerfectPlayer end

name(p::Solver) = "Perfect Player ($(round(Int, 100 * p.ϵ))% random)"

function instantiate(p::Solver, nn::MCTS.Oracle{G}) where G
  return EpsilonGreedyPlayer(PerfectPlayer(G)(), p.ϵ)
end

end
