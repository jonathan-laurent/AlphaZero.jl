# Testing remote channels and progress bars

using Distributed

addprocs(4, exeflags="--project")

@everywhere using ProgressMeter

function main()
  n = 100
  progress = Progress(100)
  chan = RemoteChannel(()->Channel{Nothing}(1))
  Threads.@spawn begin
    for i in 1:n
      take!(chan)
      next!(progress)
    end
  end
  nums = pmap(1:n) do i
    sleep(0.1)
    put!(chan, nothing)
    return i
  end
  @show sum(nums)
end

main()
