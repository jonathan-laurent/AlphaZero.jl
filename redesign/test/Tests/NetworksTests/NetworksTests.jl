module NetworksTests

using Test

include("SimpleNetTests.jl")
using .SimpleNetTests

include("SimpleResNetTests.jl")
using .SimpleResNetTests

export run_neural_networks_tests

function run_neural_networks_tests()
    @testset "SimpleNet tests" run_simplenet_tests()
    @testset "SimpleResNet tests" run_simpleresnet_tests()
end

end
