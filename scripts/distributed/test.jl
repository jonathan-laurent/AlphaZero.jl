using Distributed

include("dummy_module.jl")

addprocs(2)

@everywhere include("dummy_module.jl")

# @show nworkers()
#
# @spawnat 2 DummyModule.greet()
