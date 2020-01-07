#####
##### Schedules
#####

"""
    AbstractSchedule{R}

Abstract type for a parameter schedule, which represents a function from
nonnegative integers to numbers of type `R`. Subtypes must implement the
`getindex` operator.
"""
abstract type AbstractSchedule{R} end

function Base.getindex(s::AbstractSchedule, i::Int)
  @unimplemented
end

#####
##### Piecewise linar schedule
#####

"""
    PLSchedule{R} <: AbstractSchedule{R}

Type for piecewise linear schedules.
"""
struct PLSchedule{R} <: AbstractSchedule{R}
  # We keep the internal representation simple for JSON serialization
  xs :: Vector{Int}
  ys :: Vector{R}
  function PLSchedule{R}(xs, ys) where R
    @assert !isempty(xs)
    @assert length(xs) == length(ys)
    new{R}(xs, ys)
  end
end

PLSchedule(xs, ys) = PLSchedule{eltype(ys)}(xs, ys)

PLSchedule(cst) = PLSchedule([0], [cst])

function Base.getindex(s::PLSchedule{R}, i::Int) where R
  ptidx = findlast(x -> x <= i, s.xs)
  if isnothing(ptidx)
    # We are before the first point
    return s.ys[1]
  elseif ptidx == length(s.xs)
    # We are past the last point
    return s.ys[end]
  else
    # We are between two points
    x0, y0 = s.xs[ptidx], s.ys[ptidx]
    x1, y1 = s.xs[ptidx+1], s.ys[ptidx+1]
    y = y0 + (y1 - y0) / (x1 - x0) * (i - x0)
    R <: Integer && (y = ceil(Int, y))
    return y
  end
end

function test()
  s = PLSchedule([0, 10, 20], [0, 10, 30])
  xs = [-1, 0, 2, 10, 11, 20, 25]
  ys = [0,  0, 2, 10, 12, 30, 30]
  @assert [s[x] for x in xs] == ys
end

#test()

#####
##### Step function
#####

"""
    StepSchedule{R} <: AbstractSchedule{R}

Type for step function schedules.
"""
struct StepSchedule{R} <: AbstractSchedule{R}
  start :: R
  xs :: Vector{Int}
  ys :: Vector{R}
  function StepSchedule{R}(start, xs, ys) where R
    @assert length(xs) == length(ys)
    return new{R}(start, xs, ys)
  end
end

StepSchedule(; start, change_at, values) =
  StepSchedule{typeof(start)}(start, change_at, values)

StepSchedule(cst) = StepSchedule{typeof(cst)}(cst, [], [])

function Base.getindex(s::StepSchedule, i::Int)
   idx = findlast(x -> x <= i, s.xs)
   isnothing(idx) && (return s.start)
   return s.ys[idx]
end

"""
    Cyclic Schedule{R}
"""

function CyclicSchedule(start, mid; n, xmax)
  xmid = floor(Int, xmax * n / 2)
  xend = floor(Int, xmax * n)
  return PLSchedule([1, xmid, xend], [start, mid, start])
end
