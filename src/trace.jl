#####
##### Game trace
#####

mutable struct Trace{State}
  states :: Vector{State}
  rewards :: Vector{Float32}
  Trace(init_state) = new{typeof(init_state)}([init_state], [])
end

function trace_invariant(t::Trace)
  return length(t.states) == length(t.rewards) + 1
end

function Base.push!(t::Trace, s, r)
  push!(t.states, s)
  push!(t.rewards, r)
end

function Base.length(t::Trace)
  return length(t.rewards)
end

function total_reward(t::Trace, gamma=1.)
  return sum(gamma^(i-1) * r for (i, r) in emumerate(t.rewards))
end
