module Util

export apply_temperature

import Random
import Distributions
import ThreadPools
using Distributions: Categorical

"""
    @printing_errors expr

Evaluate expression `expr` while printing any uncaught exception on `stderr`.
This is useful to avoid silent falure of concurrent tasks, as explained in
[this issue](https://github.com/JuliaLang/julia/issues/10405).
"""
macro printing_errors(expr)
  # TODO: we should be able to replace this by `Base.erroronitor` in Julia 1.7
  return quote
    try
      $(esc(expr))
    catch e
      showerror(stderr, e, catch_backtrace())
    end
  end
end

"""
    generate_update_constructor(T)

Generate a new constructor for immutable type `T` that enables copying
an existing structure while only updating a subset of its fields.

For example, given the following struct:

    struct Point
      x :: Int
      y :: Int
    end

The generated code is equivalent to:

    Point(pt; x=pt.x, y=pt.y) = Point(x, y)

** This function may be deprecated in favor of Setfield.jl in the future.**
"""
function generate_update_constructor(T)
  fields = fieldnames(T)
  Tname = Symbol(split(string(T), ".")[end])
  base = :_old_
  @assert base ∉ fields
  fields_withdef = [Expr(:kw, f, :($base.$f)) for f in fields]
  quote
    #$Tname(;$(fields...)) = $Tname($(fields...))
    $Tname($base::$Tname; $(fields_withdef...)) = $Tname($(fields...))
  end
end

"""
    fix_probvec(π)

Convert probability vector `π` to type `Vector{Float32}` and renormalize it
if necessary.

This is useful as `Distributions.isprobvec` can be picky about its
input when it does not sum to one due to numerical errors.
"""
function fix_probvec(π)
  π = convert(Vector{Float32}, π)
  s = sum(π)
  if !(s ≈ 1)
    if iszero(s)
      n = length(π)
      π = ones(Float32, n) ./ n
    else
      π ./= s
    end
  end
  @assert Distributions.isprobvec(π)
  return π
end

"""
Draw a sample from a categorical distribution represented as a probability vector.
See [`fix_probvec`](@ref).
"""
function rand_categorical(π)
  π = fix_probvec(π)
  return rand(Categorical(π))
end

"""
    apply_temperature(π, τ)

Apply temperature `τ` to probability distribution `π`.
Handle the limit case where `τ=0`.
"""
function apply_temperature(π, τ)
  if isone(τ)
    return π
  elseif iszero(τ)
    res = zeros(eltype(π), length(π))
    res[argmax(π)] = 1
    return res
  else
    res = π .^ inv(τ)
    res ./= sum(res)
    return res
  end
end

"""
Same smoothing function that is used by Tensorboard to smooth time series.
"""
function momentum_smoothing(x, μ)
  sx = similar(x)
  isempty(x) && return x
  v = x[1]
  for i in eachindex(x)
    v = μ * x[i] + (1-μ) * v
    sx[i] = v
  end
  return sx
end

"""
    cycle_iterator(iterator)

Generate an infinite cycle from an iterator.
"""
function cycle_iterator(iterator)
  return (iterator for _ in Iterators.repeated(nothing)) |> Iterators.flatten
end

#####
##### Multithreading utilities
#####

function tmap_bg(f, xs)
  nbg = Threads.nthreads() - 1
  tasks = map(enumerate(xs)) do (i, x)
    tid = nbg > 0 ? 2 + ((i - 1) % nbg) : 1
    ThreadPools.@tspawnat tid @printing_errors f(x)
  end
  return fetch.(tasks)
end

# TODO: the `mapreduce` function should ultimately be removed and replaced by a
# more sandard solution.

"""
    mapreduce(make_worker, args, num_workers, combine, init)

In spirit, this computes `reduce(combine, map(f, args); init)` on `num_workers`
where `f` is defined below.

The `make_worker` function must create a worker `w` with two fields:
  - a `process` function such that `w.process(x)` evaluates to `f(x)`
  - a `terminate` function to be called with no argument when the worker terminates.

This function only spawns workers on background threads (with id greater or equal than 1).

!!! note

    This function makes one call to `combine` per computed element
    and so it should only be used when the associated synchronization cost
    is small compared to the cost of computing individual elements.
"""
function mapreduce(make_worker, args, num_workers, combine, init)
  next = 1
  ret = init
  lock = ReentrantLock()
  tasks = []
  nbg = Threads.nthreads() - 1
  for i in 1:num_workers
    tid = nbg > 0 ? 2 + ((i - 1) % nbg) : 1
    task = ThreadPools.@tspawnat tid Util.@printing_errors begin
      local k = 0
      worker = make_worker()
      while true
        Base.lock(lock)
        if next > length(args)
          Base.unlock(lock)
          break
        end
        k = next
        next += 1
        Base.unlock(lock)
        y = worker.process(args[k])
        Base.lock(lock)
        ret = combine(ret, y)
        Base.unlock(lock)
      end
      worker.terminate()
    end
    push!(tasks, task)
  end
  wait.(tasks)
  return ret
end

# Same semantics than mapreduce but sequential: to be used for
# debugging purposes only.
function mapreduce_sequential(make_worker, args, num_workers, combine, unit)
  ys = map(args) do x
    worker = f()
    y = worker.process(x)
    worker.terminate()
    return y
  end
  return reduce(combine, ys, init=unit)
end

macro tspawn_main(e)
  return :(ThreadPools.@tspawnat 1 $(esc(e)))
end

end