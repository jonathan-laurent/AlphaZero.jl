#####
##### Utilities to benchmark an AlphaZero system
#####

module Benchmark

import ..Util.@unimplemented
import ..Env, ..AbstractPlayer, ..AbstractNetwork
import ..MCTS, ..MctsParams, ..MctsPlayer, ..pit
import ..MinMax, ..GI

using ProgressMeter

struct DuelOutcome
  player :: String
  baseline :: String
  avgz :: Float64
  games :: Vector{Float64}
  time :: Float64
end

const Report = Vector{DuelOutcome}

abstract type Player end

function instantiate(player::Player, nn)
  @unimplemented
end

function name(::Player) :: String
  @unimplemented
end

struct Duel
  num_games :: Int
  reset_every :: Int
  player :: Player
  baseline :: Player
  function Duel(player, baseline; num_games, reset_every=0)
    return new(num_games, reset_every, player, baseline)
  end
end

function run(env::Env, duel::Duel, progress=nothing)
  player = instantiate(duel.player, env.bestnn)
  baseline = instantiate(duel.baseline, env.bestnn)
  let games = Vector{Float64}(undef, duel.num_games)
    avgz, time = @timed begin
      pit(baseline, player, duel.num_games, duel.reset_every) do i, z
        games[i] = z
        isnothing(progress) || next!(progress)
      end
    end
    return DuelOutcome(
      name(duel.player), name(duel.baseline), avgz, games, time)
  end
end

#####
##### Standard players
#####

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

struct Full <: Player
  params :: MctsParams
end

name(::Full) = "AlphaZero"

instantiate(p::Full, nn) = MctsPlayer(nn, p.params)

struct NetworkOnly <: Player
  params :: MctsParams
end

name(::NetworkOnly) = "Network Only"

function instantiate(p::NetworkOnly, nn)
  params = MctsParams(p.params, num_iters_per_turn=0)
  return MctsPlayer(nn, params)
end

# Also implements the AlphaZero.AbstractPlayer interface
struct MinMaxTS <: Player
  depth :: Int
  MinMaxTS(;depth) = new(depth)
end

struct MinMaxPlayer{G} <: AbstractPlayer{G}
  depth :: Int
end

name(p::MinMaxTS) = "MinMax (depth $(p.depth))"

function instantiate(p::MinMaxTS, nn::AbstractNetwork{G}) where G
  return MinMaxPlayer{G}(p.depth)
end

import ..reset!, ..think

reset!(::MinMaxPlayer) = nothing

function think(p::MinMaxPlayer, state, turn)
  actions = GI.available_actions(state)
  aid = MinMax.minmax(state, actions, p.depth)
  π = zeros(size(actions))
  π[aid] = 1.
  return actions[aid], π
end

end
