module Tests

using Reexport
using Test

export run_all_tests

include("Common/Common.jl")
@reexport using .Common

include("BatchedEnvsTests.jl")
@reexport using .BatchedEnvsTests

include("BitwiseTicTacToeTests.jl")
@reexport using .BitwiseTicTacToeTests

include("BitwiseConnectFourTests.jl")
@reexport using .BitwiseConnectFourTests

include("UtilTests.jl")
@reexport using .UtilTests

include("SimpleMctsTests.jl")
@reexport using .SimpleMctsTests

include("BatchedMctsTests.jl")
@reexport using .BatchedMctsTests

include("BatchedMctsAosTests.jl")
@reexport using .BatchedMctsAosTests

function run_all_tests()
    @testset "RLZero tests" begin
        run_util_tests()
        run_bitwise_tictactoe_tests()
        run_bitwise_connect_four_tests()
        run_mcts_tests()
        run_batched_mcts_tests()
    end
    return nothing
end

end
