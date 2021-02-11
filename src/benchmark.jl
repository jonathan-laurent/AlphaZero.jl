"""
Utilities to evaluate players against one another.

Typically, between each training iteration, different players
that possibly depend on the current neural network
compete against a set of baselines.
"""
module Benchmark

using ..AlphaZero

using ProgressMeter
using Statistics: mean
using Base: @kwdef

"""
    Benchmark.Player

Abstract type to specify a player that can be featured in a benchmark duel.

Subtypes must implement the following functions:
  - `Benchmark.instantiate(player, nn)`: instantiate the player specification
      into an [`AbstractPlayer`](@ref) given a neural network
  - `Benchmark.name(player)`: return a `String` describing the player
"""
abstract type Player end

# instantiate(::Player, gspec, nn)
function instantiate end

# name(::Player) :: String
function name end

"""
    Evaluation

Abstract type for a benchmark item specification.
"""
abstract type Evaluation end


"""
    Single <: Evaluation

Evaluating a single player in a one-player game.
"""
@kwdef struct Single <: Evaluation
  player :: Player
  sim :: SimParams
end


"""
    Duel <: Evaluation

Evaluating a player by pitting it against a baseline player in a two-player game.
"""
@kwdef struct Duel <: Evaluation
  player :: Player
  baseline :: Player
  sim :: SimParams
end

name(s::Single) = name(s.player)

name(d::Duel) = "$(name(d.player)) / $(name(d.baseline))"

"""
    Benchmark.run(env::Env, duel::Benchmark.Evaluation, progress=nothing)

Run a benchmark duel and return a [`Report.Evaluation`](@ref).

If a `progress` is provided, `next!(progress)` is called
after each simulated game.
"""
function run end

function run(env::Env, eval::Evaluation, progress=nothing)
  net() = Network.copy(env.bestnn, on_gpu=eval.sim.use_gpu, test_mode=true)
  if isa(eval, Single)
    simulator = Simulator(net, record_trace) do net
      instantiate(eval.player, env.gspec, net)
    end
  else
    @assert isa(eval, Duel)
    simulator = Simulator(net, record_trace) do net
      player = instantiate(eval.player, env.gspec, net)
      baseline = instantiate(eval.baseline, env.gspec, net)
      return TwoPlayers(player, baseline)
    end
  end
  samples, elapsed = @timed simulate(
    simulator, env.gspec, eval.sim,
    game_simulated=(() -> next!(progress)))
  gamma = env.params.self_play.mcts.gamma
  rewards, redundancy = rewards_and_redundancy(samples, gamma=gamma)
  return Report.Evaluation(
    name(eval), mean(rewards), redundancy, rewards, nothing, elapsed)
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

function TernaryOutcomeStatistics(report::Report.Evaluation)
  return TernaryOutcomeStatistics(report.rewards)
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

function instantiate(p::MctsRollouts, gspec::AbstractGameSpec, nn)
  return MctsPlayer(gspec, MCTS.RolloutOracle(gspec), p.params)
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

function instantiate(p::Full, gspec::AbstractGameSpec, nn)
  return MctsPlayer(gspec, nn, p.params)
end

"""
    Benchmark.NetworkOnly(;τ=1.0) <: Benchmark.Player

Player that uses the policy output by the learnt network directly,
instead of relying on MCTS.
"""
@kwdef struct NetworkOnly <: Player
  τ :: Float64 = 1.0
end

name(::NetworkOnly) = "Network Only"

function instantiate(p::NetworkOnly, ::AbstractGameSpec, nn)
  player = NetworkPlayer(nn)
  return PlayerWithTemperature(player, ConstSchedule(p.τ))
end

"""
    Benchmark.MinMaxTS(;depth, τ=0.) <: Benchmark.Player

Minmax baseline, which relies on [`MinMax.Player`](@ref).
"""
@kwdef struct MinMaxTS <: Player
  depth :: Int
  amplify_rewards :: Bool
  τ :: Float64 = 0.
end

name(p::MinMaxTS) = "MinMax (depth $(p.depth))"

function instantiate(p::MinMaxTS, ::AbstractGameSpec, nn)
  return MinMax.Player(
    depth=p.depth, amplify_rewards=p.amplify_rewards, τ=p.τ)
end

end
