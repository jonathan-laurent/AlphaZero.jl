#####
##### Memory Buffer:
##### datastructure to collect self-play experience
#####

struct TrainingExample{Board}
  b :: Board
  π :: Vector{Float64}
  z :: Float64
  n :: Int # number of times the board position `b` has been seen
  t :: Float64 # average time before game end
end

mutable struct MemoryBuffer{Board}
  # State-policy pairs accumulated during the current game.
  # The z component is 1 if white was playing and -1 otherwise
  cur :: Vector{TrainingExample{Board}}
  # Long-term memory
  mem :: CircularBuffer{TrainingExample{Board}}
  last_batch_size :: Int
  function MemoryBuffer{B}(size) where B
    new{B}([], CircularBuffer{TrainingExample{B}}(size), 0)
  end
end

Base.length(b::MemoryBuffer) = length(b.mem)

function merge_samples(es::Vector{TrainingExample{B}}) where B
  b = es[1].b
  π = mean(e.π for e in es)
  z = mean(e.z for e in es)
  n = sum(e.n for e in es)
  t = mean(e.t for e in es)
  return TrainingExample{B}(b, π, z, n, t)
end

# Merge samples that correspond to identical boards
function merge_by_board(es::Vector{TrainingExample{B}}) where B
  dict = Dict{B, Vector{TrainingExample{B}}}()
  sizehint!(dict, length(es))
  for e in es
    if haskey(dict, e.b)
      push!(dict[e.b], e)
    else
      dict[e.b] = [e]
    end
  end
  return [merge_samples(es) for es in values(dict)]
end

get(buf::MemoryBuffer) = buf.mem[:]

last_batch(buf::MemoryBuffer) = buf.mem[end-buf.last_batch_size+1:end]

last_batch_size(buf::MemoryBuffer) = buf.last_batch_size

function push_sample!(buf::MemoryBuffer, board, policy, white_playing, turn)
  player_code = white_playing ? 1. : -1.
  ex = TrainingExample(board, policy, player_code, 1, float(turn))
  push!(buf.cur, ex)
  buf.last_batch_size += 1
end

function push_game!(buf::MemoryBuffer, white_reward, game_length)
  for ex in buf.cur
    r = ex.z * white_reward
    t = game_length - ex.t
    push!(buf.mem, TrainingExample(ex.b, ex.π, r, ex.n, t))
  end
  empty!(buf.cur)
end

function new_batch!(buf::MemoryBuffer)
  buf.last_batch_size = 0
end
