################################################################################
# Generic utilities
################################################################################

module Util

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

weighted_mse(ŷ, y, w) = sum((ŷ .- y).^2 .* w) * 1 // length(y)

end

################################################################################
