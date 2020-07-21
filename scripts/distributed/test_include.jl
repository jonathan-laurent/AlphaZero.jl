using Distributed

include("include_workaround.jl")

addprocs(2)

include_everywhere("dummy_module.jl")

@everywhere using .DummyModule

@show nworkers()

greet()

@spawnat 2 greet()
