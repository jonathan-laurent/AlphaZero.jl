module Tests

using Reexport
using Test

export run_all_tests

include("Common/Common.jl")
@reexport using .Common

include("UtilTests.jl")
@reexport using .UtilTests: run_util_tests

include("NetworksTests/NetworksTests.jl")
@reexport using .NetworksTests: run_neural_networks_tests

include("EnvTests/EnvTests.jl")
@reexport using .EnvTests: run_env_tests

include("MctsTests/MctsTests.jl")
@reexport using .MctsTests: run_mcts_tests

function run_all_tests()
    @testset "RLZero tests" begin
        @testset "utility tests" run_util_tests()
        @testset "neural network tests" run_neural_networks_tests()
        @testset "env tests" run_env_tests()
        @testset "mcts tests" run_mcts_tests()
    end
    return nothing
end

end
