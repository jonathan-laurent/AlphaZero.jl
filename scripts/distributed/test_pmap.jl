using Distributed

Distributed.addprocs(4)

@everywhere power(x, i) = x ^ i

function compute_series()
  x = 0.5
  @show x
  powers = pmap(0:100) do i
    return power(x, i)
  end
  @show sum(powers)
end

compute_series()
