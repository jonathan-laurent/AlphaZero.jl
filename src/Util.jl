module Util

export Option

import Random

const Option{T} = Union{T, Nothing}

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

infinity(::Type{R}) where R <: Real = one(R) / zero(R)

function batches(X, batchsize)
  n = size(X, 2)
  b = batchsize
  nbatches = n ÷ b
  return (X[:,(1+b*(i-1)):(b*i)] for i in 1:nbatches)
end

function random_batches(xs::Tuple, batchsize)
  let n = size(xs[1], 2)
  let perm = Random.randperm(n)
  bxs = map(xs) do x
    batches(x[:,perm], batchsize)
  end
  zip(bxs...)
  end end
end

end
