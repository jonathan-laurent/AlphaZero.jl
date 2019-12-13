module Util

export Option, @unimplemented

import Random

const Option{T} = Union{T, Nothing}

struct Unimplemented <: Exception end

macro unimplemented()
  return quote
    throw(Unimplemented())
  end
end

# concat_cols(cols) == hcat(cols...)
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

# superpose(xs) == cat(xs..., dims=ndims(first(xs))+1)
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

infinity(::Type{R}) where R <: Real = one(R) / zero(R)

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

function periodic_gc(gc, batchs, period)
  n = 0
  function process(b)
    n += size(b[1])[end]
    if period > 0 && n >= period
      n = 0
      gc()
    end
    return b
  end
  period > 0 && gc()
  return (process(b) for b in batchs)
end

# Print uncaught exceptions
# In response to: https://github.com/JuliaLang/julia/issues/10405
macro printing_errors(expr)
  return quote
    try
      $(esc(expr))
    catch e
      showerror(stderr, e, catch_backtrace())
    end
  end
end

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

end
