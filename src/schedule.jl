#####
##### Schedules
#####

"""
    AbstractSchedule{R}

Abstract type for a parameter schedule, which represents a function from
nonnegative integers to numbers of type `R`. Subtypes must implement the
`getindex(s::AbstractSchedule, i::Int)` operator.
"""
abstract type AbstractSchedule{R} end

#####
##### Constant Schedule
#####

struct ConstSchedule{R} <: AbstractSchedule{R}
  value :: R
end

Base.getindex(s::ConstSchedule, i::Int) = s.value

Base.convert(::Type{ConstSchedule{R}}, x::R) where {R <: Number} =
  ConstSchedule(x)

#####
##### Piecewise linar schedule
#####

"""
    PLSchedule{R} <: AbstractSchedule{R}

Type for piecewise linear schedules.

# Constructors

    PLSchedule(cst)

Return a schedule with a constant value `cst`.

    PLSchedule(xs, ys)

Return a piecewise linear schedule such that:
  - For all `i`, `(xs[i], ys[i])` belongs to the schedule's graph.
  - Before `xs[1]`, the schedule has value `ys[1]`.
  - After `xs[end]`, the schedule has value `ys[end]`.
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

# Constructor

    StepSchedule(;start, change_at, values)

Return a schedule that has initial value `start`. For all `i`, the schedule
takes value `values[i]` at step `change_at[i]`.
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

function Base.getindex(s::StepSchedule, i::Int)
   idx = findlast(x -> x <= i, s.xs)
   isnothing(idx) && (return s.start)
   return s.ys[idx]
end

"""
    CyclicSchedule(base, mid, term; n, xmid=0.45, xback=0.90)

Return the [`PLSchedule`](@ref) that is typically used for cyclic
learning rate scheduling.
"""
function CyclicSchedule(base, mid, term; n, xmid=0.45, xback=0.90)
  nmid  = floor(Int, xmid * n)
  nback = floor(Int, xback * n)
  return PLSchedule([1, nmid, nback, n], [base, mid, base, term])
end
