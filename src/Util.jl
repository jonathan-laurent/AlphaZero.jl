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

function batches(X, batchsize)
  n = size(X)[end]
  b = batchsize
  nbatches = n ÷ b
  return (selectdim(X, ndims(X), (1+b*(i-1)):(b*i)) for i in 1:nbatches)
end

function random_batches(xs::Tuple, batchsize)
  let n = size(xs[1])[end]
  let perm = Random.randperm(n)
  bxs = map(xs) do x
    batches(selectdim(x, ndims(x), perm), batchsize)
  end
  zip(bxs...)
  end end
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

end
