module Util

export Option, apply_temperature

import Random
using Distributions: Categorical

const Option{T} = Union{T, Nothing}

infinity(::Type{R}) where R <: Real = one(R) / zero(R)

"""
    concat_columns(cols) == hcat(cols...) # but faster
"""
function concat_columns(cols)
  @assert !isempty(cols)
  nsamples = length(cols)
  excol = first(cols)
  sdim = length(excol)
  arr = similar(excol, (sdim, nsamples))
  for (i, col) in enumerate(cols)
    arr[:,i] = col
  end
  return arr
end

"""
    superpose(xs) == cat(xs..., dims=ndims(first(xs))+1) # but faster
"""
function superpose(arrays)
  n = length(arrays)
  @assert n > 0
  ex = first(arrays)
  dest = similar(ex, size(ex)..., n)
  i = 1
  for src in arrays
    for j in eachindex(src)
      dest[i] = src[j]
      i += 1
    end
  end
  return dest
end

"""
    @printing_errors expr

Evaluate expression `expr` while printing any uncaught exception on `stderr`.

This is useful to avoid silent falure of concurrent tasks, as explained in
[this issue](https://github.com/JuliaLang/julia/issues/10405).
"""
macro printing_errors(expr)
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
Same smoothing function that is used by Temsorboard to smooth time series.
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
    batches(X, batchsize; partial=false)

Take a data tensor `X` and split it into batches of fixed size along the
last dimension of `X`.

If `partial=true` and the number of samples in `X` is
not a multiple of `batchsize`, then an additional smaller batch is added
at the end (otherwise, it is discarded).
"""
function batches(X, batchsize; partial=false)
  n = size(X)[end]
  b = batchsize
  nbatches = n ÷ b
  # The call to `copy` after selectdim is important because Flux does not
  # deal well with views.
  select(a, b) = copy(selectdim(X, ndims(X), a:b))
  batches = [select(1+b*(i-1), b*i) for i in 1:nbatches]
  if partial && n % b > 0
    # If the number of samples is not a multiple of the batch size
    push!(batches, select(b*nbatches+1, n))
  end
  return batches
end

function batches_tests()
  @assert batches(collect(1:5), 2, partial=true) == [[1, 2], [3, 4], [5]]
end

"""
    random_batches(convert, data::Tuple, batchsize; partial=false)

Take a tuple of data tensors, shuffle its samples according to a random
permutation and split them into a sequence of minibatches.

The result is a lazy iterator that calls `convert` on the tensors of each
new batch right before returning it. The `convert` function is typically
used to transfer data onto the GPU.

!!! note
    In the future, it may be good to deprecate this function along with
    [`random_batches_stream`](@ref) and use a standard solution instead,
    such as `Flux.DataLoader`.
"""
function random_batches(
  convert, data::Tuple, batchsize; partial=false)
  n = size(data[1])[end]
  perm = Random.randperm(n)
  batchs = map(data) do x
    batches(selectdim(x, ndims(x), perm), batchsize, partial=partial)
  end
  batchs = collect(zip(batchs...))
  return (convert.(b) for b in batchs)
end

"""
    random_batches_stream(convert, data::Tuple, batchsize)

Generate an infinite stateful iterator of random batches by calling
[`random_batches`](@ref) repeatedly. Every sample is guaranteed to be drawn
exactly once per epoch.
"""
function random_batches_stream(convert, data::Tuple, batchsize)
  partial = size(data[1])[end] < batchsize
  return Iterators.Stateful(Iterators.flatten((
    random_batches(convert, data, batchsize, partial=partial)
    for _ in Iterators.repeated(nothing))))
end

#####
##### Multithreading utilities
#####

function threads_pmap(f, xs)
  return fetch.([Threads.@spawn @printing_errors f(x) for x in xs])
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
  for w in 1:num_workers
    task = Threads.@spawn Util.@printing_errors begin
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

end
