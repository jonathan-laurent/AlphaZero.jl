module NetworksTests

using Test

include("SimpleNetTests.jl")
using .SimpleNetTests

export run_neural_networks_tests

function run_neural_networks_tests()
    @testset "SimpleNet tests" run_simplenet_tests()
    return nothing
end

end
