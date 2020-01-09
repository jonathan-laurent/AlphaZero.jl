#####
##### Memory Buffer:
##### datastructure to collect self-play experience
#####

"""
    TrainingSample{Board}

Type of a training sample. A sample features the following fields:
- `b::Board` is the board position (by convention, white is to play)
- `π::Vector{Float64}` is the recorded MCTS policy for this position
- `z::Float64` is the reward collected at the end of the game
- `t::Float64` is the number of moves remaining before the end of the game
- `n::Int` is the number of times the board position `b` was recorded

As revealed by the last field `n`, several samples that correspond to the
same board position can be merged, in which case the `π`, `z` and `t`
fields are averaged together.
"""
struct TrainingSample{Board}
  b :: Board
  π :: Vector{Float64}
  z :: Float64
  t :: Float64 # average time before game end
  n :: Int # number of times the board position `b` has been seen
end

"""
    MemoryBuffer{Board}

A circular buffer to hold memory samples.

# How to use

- Use `new_batch!(mem)` to start a new batch, typically once per iteration
  before self-play.
- Use `push_sample!(mem, board, policy, white_playing, turn)` to record
  a sample during a game, where `turn` is the number of actions that have
  been played by both players since the start of the game.
- Use `push_game!(mem, white_reward, game_length)` when a game terminates
  for which samples have been collected.
"""
mutable struct MemoryBuffer{Board}
  # State-policy pairs accumulated during the current game.
  # The z component is 1 if white was playing and -1 otherwise
  cur :: Vector{TrainingSample{Board}}
  # Long-term memory
  mem :: CircularBuffer{TrainingSample{Board}}
  last_batch_size :: Int
  function MemoryBuffer{B}(size, experience=[]) where B
    mem = CircularBuffer{TrainingSample{B}}(size)
    append!(mem, experience)
    new{B}([], mem, 0)
  end
end

function Base.empty!(b::MemoryBuffer)
  empty!(b.cur)
  empty!(b.mem)
  b.last_batch_size = 0
end

Base.length(b::MemoryBuffer) = length(b.mem)

function merge_samples(es::Vector{TrainingSample{B}}) where B
  b = es[1].b
  π = mean(e.π for e in es)
  z = mean(e.z for e in es)
  n = sum(e.n for e in es)
  t = mean(e.t for e in es)
  return TrainingSample{B}(b, π, z, t, n)
end

# Merge samples that correspond to identical boards
function merge_by_board(es::AbstractVector{TrainingSample{B}}) where B
  dict = Dict{B, Vector{TrainingSample{B}}}()
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

last_batch(buf::MemoryBuffer) = buf.mem[end-last_batch_size(buf)+1:end]

last_batch_size(buf::MemoryBuffer) = min(buf.last_batch_size, length(buf))

function push_sample!(buf::MemoryBuffer, board, policy, white_playing, turn)
  player_code = white_playing ? 1. : -1.
  ex = TrainingSample(board, policy, player_code, float(turn), 1)
  push!(buf.cur, ex)
  buf.last_batch_size += 1
end

function push_game!(buf::MemoryBuffer, white_reward, game_length)
  for ex in buf.cur
    r = ex.z * white_reward
    t = game_length - ex.t
    push!(buf.mem, TrainingSample(ex.b, ex.π, r, t, ex.n))
  end
  empty!(buf.cur)
end

function new_batch!(buf::MemoryBuffer)
  buf.last_batch_size = 0
end
