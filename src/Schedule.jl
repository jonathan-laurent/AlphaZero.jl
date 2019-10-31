#####
##### Schedules
#####

abstract type AbstractSchedule{R} end

function get(s::AbstractSchedule, i)
  #@unimplemented
end

#####
##### Piecewise linar schedule
#####

# We keep the internal representation simple for JSON serialization
struct PLSchedule{R}
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

function get(s::PLSchedule{R}, i) where R
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
  @assert [get(s, x) for x in xs] == ys
end

test()
