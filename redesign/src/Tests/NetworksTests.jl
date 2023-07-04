module NetworksTests

include("SimpleNetTests.jl")
using .SimpleNetTests

using Test


export run_neural_networks_tests


function run_neural_networks_tests()
    @testset "SimpleNet tests" run_simplenet_tests()
    return nothing
end

end
