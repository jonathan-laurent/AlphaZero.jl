using Test

import AlphaZero: Util

n = 1000
num_workers = 100
res = Util.mapreduce(1:n, num_workers, +, 0) do
  process(i) = i^2
  terminate() = nothing
  return (process=process, terminate=terminate)
end
@test res == sum(i^2 for i in 1:n)