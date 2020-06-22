#####
##### Game trace
#####

mutable struct Trace{Game, State}
  states :: Vector{State}
  policies :: Vector{Vector{Float64}}
  rewards :: Vector{Float32}
  function Trace{G}(init_state) where G
    return new{G, GI.State(G)}([init_state], [], [])
  end
end

GameType(::Trace{Game}) where Game = Game

function trace_invariant(t::Trace)
  return length(t.policies) == length(t.rewards) == length(states) - 1
end

function Base.push!(t::Trace, s, π, r)
  push!(t.states, s)
  push!(t.policies, π)
  push!(t.rewards, r)
end

function Base.length(t::Trace)
  return length(t.rewards)
end

function total_reward(t::Trace, gamma=1.)
  return sum(gamma^(i-1) * r for (i, r) in enumerate(t.rewards))
end
