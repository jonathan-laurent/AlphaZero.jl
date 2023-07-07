module Tests

using Reexport
using Test

export run_all_tests

include("Common/Common.jl")
@reexport using .Common

include("NetworksTests.jl")
@reexport using .NetworksTests

include("BatchedEnvsTests.jl")
@reexport using .BatchedEnvsTests

include("EnvTests/BitwiseRandomWalk1DTests.jl")
@reexport using .BitwiseRandomWalk1DTests

include("EnvTests/BitwiseTicTacToeTests.jl")
@reexport using .BitwiseTicTacToeTests

include("EnvTests/BitwiseConnectFourTests.jl")
@reexport using .BitwiseConnectFourTests

include("UtilTests.jl")
@reexport using .UtilTests

include("MctsTests/SimpleMctsTests.jl")
@reexport using .SimpleMctsTests

include("MctsTests/BatchedMctsTests.jl")
@reexport using .BatchedMctsTests

include("MctsTests/BatchedMctsAosTests.jl")
@reexport using .BatchedMctsAosTests

function run_all_tests()
    @testset "RLZero tests" begin
        @testset "utility tests" run_util_tests()
        @testset "neural network tests" run_neural_networks_tests()
        @testset "random walk env tests" run_bitwise_random_walk_tests()
        @testset "tictactoe tests" run_bitwise_tictactoe_tests()
        @testset "connect-4 tests" run_bitwise_connect_four_tests()
        @testset "simple mcts tests" run_simple_mcts_tests()
        @testset "batched mcts SoA tests" run_batched_mcts_tests()
        # @testset "batched mcts AoS tests" run_batched_mcts_aos_tests()
    end
    return nothing
end

end
