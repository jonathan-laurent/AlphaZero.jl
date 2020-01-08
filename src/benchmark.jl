"""
Utilities to evaluate players against one another.
"""
module Benchmark

import ..Util.@unimplemented
import ..Env, ..AbstractPlayer, ..AbstractNetwork, ..NetworkPlayer
import ..MCTS, ..MctsParams, ..MctsPlayer, ..pit
import ..ColorPolicy, ..ALTERNATE_COLORS
import ..MinMax, ..GI

using ProgressMeter

"""
    Benchmark.DuelOutcome

The outcome of a duel between two players.

# Fields
- `player` and `baseline` are `String` fields containing the names of
    both players involved in the duel
- `avgz` is the average reward collected by `player`
- `rewards` is a vector containing all rewards collected by `player`
    (one per game played)
- `time` is the amount of computing time took consumed by the duel, in seconds
"""
struct DuelOutcome
  player :: String
  baseline :: String
  avgz :: Float64
  rewards :: Vector{Float64}
  time :: Float64
end

"""
    Benchmark.Report = Vector{Benchmark.DuelOutcome}

A benchmark report is a vector of [`Benchmark.DuelOutcome`](@ref) objects.
"""
const Report = Vector{DuelOutcome}

"""
    Benchmark.Player

Abstract type for a player that can be featured in a benchmark duel.
"""
abstract type Player end

function instantiate(player::Player, nn)
  @unimplemented
end

function name(::Player) :: String
  @unimplemented
end

"""
    Benchmark.Duel(player, baseline; num_games)

Specify a duel that consists in `num_games` games between
`player` and `baseline`, each of them of type [`Benchmark.Player`](@ref).

# Optional keyword arguments
- `reset_every`: if set, the MCTS tree is reset every `reset_mcts_every` games
    to avoid running out of memory
- `color_policy` has type [`ColorPolicy`](@ref) (`ALTERNATE_COLORS`) by default
"""
struct Duel
  num_games :: Int
  reset_every :: Union{Nothing, Int}
  color_policy :: ColorPolicy
  player :: Player
  baseline :: Player
  function Duel(player, baseline;
      num_games, reset_every=nothing, color_policy=ALTERNATE_COLORS)
    return new(num_games, reset_every, color_policy, player, baseline)
  end
end

function run(env::Env, duel::Duel, progress=nothing)
  player = instantiate(duel.player, env.bestnn)
  baseline = instantiate(duel.baseline, env.bestnn)
  let games = Vector{Float64}(undef, duel.num_games)
    avgz, time = @timed begin
      pit(
        baseline, player, duel.num_games,
        reset_every=duel.reset_every, color_policy=duel.color_policy) do i, z
          games[i] = z
          isnothing(progress) || next!(progress)
      end
    end
    return DuelOutcome(
      name(duel.player), name(duel.baseline), avgz, games, time)
  end
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

name(::MctsRollouts) = "MCTS Rollouts"

function instantiate(p::MctsRollouts, nn::AbstractNetwork{G}) where G
  params = MctsParams(p.params,
    num_workers=1,
    use_gpu=false)
  return MctsPlayer(MCTS.RolloutOracle{G}(), params)
end

"""
    Benchmark.Full(params) <: Benchmark.Player

Full AlphaZero player that combines MCTS with the learnt network.

Argument `params` has type [`MctsParams`](@ref).
"""
struct Full <: Player
  params :: MctsParams
end

name(::Full) = "AlphaZero"

instantiate(p::Full, nn) = MctsPlayer(nn, p.params)

"""
    Benchmark.NetworkOnly(;use_gpu=true) <: Benchmark.Player

Player that uses the policy output by the learnt network directly,
instead of relying on MCTS.
"""
struct NetworkOnly <: Player
  use_gpu :: Bool
  NetworkOnly(;use_gpu=true) = new(use_gpu)
end

name(::NetworkOnly) = "Network Only"

instantiate(p::NetworkOnly, nn) = NetworkPlayer(nn, use_gpu=p.use_gpu)

"""
    Benchmark.MinMaxTS(;depth, τ=0.) <: Benchmark.Player

Minmax baseline, which relies on [`MinMax.Player`](@ref).
"""
struct MinMaxTS <: Player
  depth :: Int
  τ :: Float64
  MinMaxTS(;depth, τ=0.) = new(depth, τ)
end

name(p::MinMaxTS) = "MinMax (depth $(p.depth))"

function instantiate(p::MinMaxTS, ::AbstractNetwork{G}) where G
  return MinMax.Player{G}(depth=p.depth, τ=p.τ)
end

end
