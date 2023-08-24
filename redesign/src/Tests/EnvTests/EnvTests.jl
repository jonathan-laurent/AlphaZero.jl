module EnvTests

using Test

include("../BatchedEnvsTests.jl")
using .BatchedEnvsTests

include("BitwiseRandomWalk1DTests.jl")
using .BitwiseRandomWalk1DTests: run_bitwise_random_walk_tests

include("BitwiseTicTacToeTests.jl")
using .BitwiseTicTacToeTests: run_bitwise_tictactoe_tests

include("BitwiseConnectFourTests.jl")
using .BitwiseConnectFourTests: run_bitwise_connect_four_tests

export run_env_tests

function run_env_tests()
    @testset "random walk 1d tests" run_bitwise_random_walk_tests()
    @testset "tictactoe tests" run_bitwise_tictactoe_tests()
    @testset "connect-4 tests" run_bitwise_connect_four_tests()
end


end
