module MctsTests

using Test

include("SimpleMctsTests.jl")
using .SimpleMctsTests: run_simple_mcts_tests

include("BatchedMctsTests.jl")
using .BatchedMctsTests: run_batched_mcts_tests

include("BatchedMctsAosTests.jl")
using .BatchedMctsAosTests: run_batched_mcts_aos_tests

include("OraclesTests.jl")
using .OraclesTests: run_oracle_tests

export run_mcts_tests

function run_mcts_tests()
    @testset "oracle tests" run_oracle_tests()
    @testset "simple mcts tests" run_simple_mcts_tests()
    @testset "batched mcts SoA tests" run_batched_mcts_tests()
    @testset "batched mcts AoS tests" run_batched_mcts_aos_tests()
end


end
