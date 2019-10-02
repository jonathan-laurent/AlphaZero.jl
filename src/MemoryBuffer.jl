#####
##### Memory Buffer:
##### datastructure to collect self-play experience
#####

struct TrainingExample{Board}
  b :: Board
  π :: Vector{Float64}
  z :: Float64
  n :: Int # number of times the board position `b` has been seen
end

struct MemoryBuffer{Board}
  # State-policy pairs accumulated during the current game.
  # The z component is 1 if white was playing and -1 otherwise
  cur :: Vector{TrainingExample{Board}}
  # Long-term memory
  mem :: CircularBuffer{TrainingExample{Board}}

  function MemoryBuffer{B}(size) where B
    new{B}([], CircularBuffer{TrainingExample{B}}(size))
  end
end

Base.length(b::MemoryBuffer) = length(b.mem)

function merge_samples(es::Vector{TrainingExample{B}}) where B
  b = es[1].b
  π = mean(e.π for e in es)
  z = mean(e.z for e in es)
  n = sum(e.n for e in es)
  return TrainingExample{B}(b, π, z, n)
end

# Get memory content, merging samples for identical boards
function get(b::MemoryBuffer{B}) where B
  dict = Dict{B, Vector{TrainingExample{B}}}()
  sizehint!(dict, length(b))
  for e in b.mem[:]
    if haskey(dict, e.b)
      push!(dict[e.b], e)
    else
      dict[e.b] = [e]
    end
  end
  return [merge_samples(es) for es in values(dict)]
end

function push_sample!(buf::MemoryBuffer, board, policy, white_playing)
  player_code = white_playing ? 1.0 : -1.0
  ex = TrainingExample(board, policy, player_code, 1)
  push!(buf.cur, ex)
end

function push_game!(buf::MemoryBuffer, white_reward)
  for ex in buf.cur
    r = ex.z * white_reward
    push!(buf.mem, TrainingExample(ex.b, ex.π, r, ex.n))
  end
  empty!(buf.cur)
end
