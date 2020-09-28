#####
##### Memory Buffer:
##### Datastructure to collect self-play experience
#####

"""
    TrainingSample{State}

Type of a training sample. A sample features the following fields:
- `s::State` is the state
- `π::Vector{Float64}` is the recorded MCTS policy for this position
- `z::Float64` is the discounted reward cumulated from state `s`
- `t::Float64` is the (average) number of moves remaining before the end of the game
- `n::Int` is the number of times the state `s` was recorded

As revealed by the last field `n`, several samples that correspond to the
same state can be merged, in which case the `π`, `z` and `t`
fields are averaged together.
"""
struct TrainingSample{State}
  s :: State
  π :: Vector{Float64}
  z :: Float64
  t :: Float64
  n :: Int
end

sample_state_type(::Type{<:TrainingSample{S}}) where S = S

"""
    MemoryBuffer(game_spec, size, experience=[])

A circular buffer to hold memory samples.
"""
mutable struct MemoryBuffer{GameSpec, State}
  gspec :: GameSpec
  buf :: CircularBuffer{TrainingSample{State}}
  cur_batch_size :: Int
  function MemoryBuffer(gspec, size, experience=[])
    State = GI.state_type(gspec)
    buf = CircularBuffer{TrainingSample{State}}(size)
    append!(buf, experience)
    new{typeof(gspec), State}(gspec, buf, 0)
  end
end

"""
    get_experience(::MemoryBuffer) :: Vector{<:TrainingSample}

Return all samples in the memory buffer.
"""
get_experience(mem::MemoryBuffer) = mem.buf[:]

last_batch(mem::MemoryBuffer) = mem.buf[end-cur_batch_size(mem)+1:end]

cur_batch_size(mem::MemoryBuffer) = min(mem.cur_batch_size, length(mem))

new_batch!(mem::MemoryBuffer) = (mem.cur_batch_size = 0)

function Base.empty!(mem::MemoryBuffer)
  empty!(mem.buf)
  mem.cur_batch_size = 0
end

Base.length(mem::MemoryBuffer) = length(mem.buf)

"""
    push_trace!(mem::MemoryBuffer, trace::Trace, gamma)

Collect samples out of a game trace and add them to the memory buffer.

Here, `gamma` is the reward discount factor.
"""
function push_trace!(mem::MemoryBuffer, trace, gamma)
  n = length(trace)
  wr = 0.
  for i in reverse(1:n)
    wr = gamma * wr + trace.rewards[i]
    s = trace.states[i]
    π = trace.policies[i]
    wp = GI.white_playing(GI.init(mem.gspec, s))
    z = wp ? wr : -wr
    t = float(n - i + 1)
    push!(mem.buf, TrainingSample(s, π, z, t, 1))
  end
  mem.cur_batch_size += n
end

function merge_samples(samples)
  s = samples[1].s
  π = mean(e.π for e in samples)
  z = mean(e.z for e in samples)
  n = sum(e.n for e in samples)
  t = mean(e.t for e in samples)
  return eltype(samples)(s, π, z, t, n)
end

# Merge samples that correspond to identical states
function merge_by_state(samples)
  Sample = eltype(samples)
  State = sample_state_type(Sample)
  dict = Dict{State, Vector{Sample}}()
  sizehint!(dict, length(samples))
  for s in samples
    if haskey(dict, s.s)
      push!(dict[s.s], s)
    else
      dict[s.s] = [s]
    end
  end
  return [merge_samples(ss) for ss in values(dict)]
end

function apply_symmetry(gspec, sample, (symstate, aperm))
  mask = GI.actions_mask(GI.init(gspec, sample.s))
  symmask = GI.actions_mask(GI.init(gspec, symstate))
  π = zeros(eltype(sample.π), length(mask))
  π[mask] = sample.π
  π = π[aperm]
  @assert iszero(π[.~symmask])
  π = π[symmask]
  return typeof(sample)(
    symstate, π, sample.z, sample.t, sample.n)
end

function augment_with_symmetries(gspec, samples)
  symsamples = [apply_symmetry(gspec, s, sym)
    for s in samples for sym in GI.symmetries(gspec, s.s)]
  return [samples ; symsamples]
end
