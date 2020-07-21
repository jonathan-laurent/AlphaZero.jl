using Distributed

# This fix only works on a single machine
function include_everywhere(filepath)
    fullpath = joinpath(@__DIR__, filepath)
    @sync for p in procs()
        @async remotecall_wait(include, p, fullpath)
    end
end

addprocs(2)

include_everywhere("dummy_module.jl")

@everywhere using .DummyModule

@show nworkers()

greet()

@spawnat 2 greet()
